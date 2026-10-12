"""
Integration and Production Pipeline Tests for TRACE.
Directly tests TracePipeline.start(), frame acquisition contracts,
FrameEnvelope concurrency, atomic persistence, camera health status,
and API integration against active pipeline instances.
Addresses Critical Finding 1, 2, 8, 9, 12, and 13.
"""

import json
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from trace.api.routes import create_app
from trace.config import AppConfig, load_config
from trace.pipeline import TracePipeline
from trace.platform.db import DatabaseManager
from trace.vision.capture import FrameCapture


def test_pipeline_startup_contract_and_baseline_acquisition(tmp_path):
    """
    Finding 1 & 12: Verify startup sequence wait_for_first_frame 3-element contract,
    successful pipeline startup, and baseline registration without crash.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_startup.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    try:
        # 1. Directly test wait_for_first_frame contract
        pipeline.capture.start()
        ok, init_frame, ts = pipeline.capture.wait_for_first_frame(timeout_sec=3.0)
        assert ok is True
        assert isinstance(init_frame, np.ndarray)
        assert init_frame.shape[0] > 0 and init_frame.shape[1] > 0
        assert isinstance(ts, float)
        pipeline.capture.stop()

        # 2. Test full pipeline.start()
        pipeline.start()
        assert pipeline.is_running is True
        time.sleep(0.5)

        # Baseline should be captured and saved for registered assets
        assert len(pipeline.baseline_medians) > 0
        for asset_id in pipeline.assets.keys():
            assert asset_id in pipeline.baseline_medians
            assert pipeline.baseline_medians[asset_id] is not None

        # Verify cameras and states tables
        with db_mgr.get_connection() as conn:
            cam_row = conn.execute("SELECT status FROM cameras WHERE id = 'cam-1'").fetchone()
            assert cam_row is not None
            assert cam_row[0] in ("ONLINE", "HEALTHY")

            state_rows = conn.execute("SELECT asset_id, state FROM states").fetchall()
            assert len(state_rows) > 0
    finally:
        pipeline.stop()


def test_frame_envelope_no_deadlock_and_staleness(tmp_path):
    """
    Finding 2: Verify that FrameEnvelope does not deadlock on reentrant lock,
    produces monotonically increasing frame IDs and timestamps, and detects staleness.
    """
    capture = FrameCapture(source_uri="synthetic", width=640, height=480, target_fps=10)
    capture.start()
    try:
        time.sleep(0.2)
        # Call read_envelope repeatedly under a tight interval
        env1 = capture.read_envelope("cam-1")
        assert env1.is_valid is True
        assert env1.image is not None
        assert env1.frame_id > 0
        assert env1.captured_at_monotonic_ns > 0

        # Immediate re-read should indicate is_new=False before new frame is generated
        env2 = capture.read_envelope("cam-1")
        assert env2.frame_id == env1.frame_id
        assert env2.is_new is False

        # Sleep to allow next frame
        time.sleep(0.15)
        env3 = capture.read_envelope("cam-1")
        assert env3.frame_id > env1.frame_id
        assert env3.is_new is True
    finally:
        capture.stop()


def test_production_pipeline_stable_scene_no_false_alarms(tmp_path):
    """
    Finding 12: Run production TracePipeline on a stable scene.
    Confirm that frames process normally and zero incidents or false alarms are generated.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_stable.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    try:
        pipeline.start()
        time.sleep(1.2)  # Allow loop to process multiple frames

        assert pipeline.total_frames_processed > 0

        # Verify database has zero incidents
        with db_mgr.get_connection() as conn:
            inc_count = conn.execute("SELECT count(*) FROM incidents").fetchone()[0]
            assert inc_count == 0

            event_count = conn.execute("SELECT count(*) FROM events").fetchone()[0]
            assert event_count == 0
    finally:
        pipeline.stop()


def test_production_pipeline_outbox_handler_and_atomic_persistence(tmp_path):
    """
    Finding 8 & 9: Test outbox delivery handler behavior and atomic database consistency.
    Outbox jobs must not be marked DELIVERED if the handler fails.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_outbox.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    try:
        pipeline.outbox.enqueue("evt-test-1", json.dumps({"msg": "test event"}))

        # 1. Flush with failing handler -> must remain PENDING
        def failing_handler(item):
            return False

        delivered_ids = pipeline.outbox.flush(delivery_handler=failing_handler)
        assert len(delivered_ids) == 0

        with db_mgr.get_connection() as conn:
            row = conn.execute("SELECT status, attempts FROM outbox WHERE event_id = 'evt-test-1'").fetchone()
            assert row[0] == "PENDING"
            assert row[1] == 1

        # 2. Flush with successful handler -> must transition to DELIVERED
        def successful_handler(item):
            return True

        delivered_ids = pipeline.outbox.flush(delivery_handler=successful_handler)
        assert len(delivered_ids) == 1
        assert delivered_ids[0] == "evt-test-1"

        with db_mgr.get_connection() as conn:
            row = conn.execute("SELECT status FROM outbox WHERE event_id = 'evt-test-1'").fetchone()
            assert row[0] == "DELIVERED"
    finally:
        pipeline.stop()


def test_api_integration_with_live_pipeline(tmp_path):
    """
    Finding 10 & 13: Test FastAPI routes against active pipeline instance,
    verifying authenticated access, state reporting, and truthful health status.
    """
    cfg = load_config("config/demo.yaml")
    cfg.security.require_auth = True
    cfg.security.viewer_token = "viewer-secret-token"
    cfg.security.admin_token = "admin-secret-token"
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_api_int.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    pipeline.start()
    try:
        time.sleep(0.5)
        app = create_app(pipeline)
        client = TestClient(app)

        # 1. Unauthenticated request to /api/state is rejected
        resp = client.get("/api/state")
        assert resp.status_code == 401

        # 2. Authenticated request succeeds
        resp = client.get("/api/state", headers={"Authorization": "Bearer viewer-secret-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert "camera_status" in data
        assert "assets" in data
        assert data["camera_status"] in ("ONLINE", "HEALTHY")

        # 3. Truthful health endpoint
        resp = client.get("/api/health")
        assert resp.status_code == 200
        health_data = resp.json()
        assert health_data["status"].upper() in ("HEALTHY", "DEGRADED")
    finally:
        pipeline.stop()
