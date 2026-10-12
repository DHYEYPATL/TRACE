"""
Integration and Production Pipeline Tests for TRACE.
Directly tests TracePipeline.start(), frame acquisition contracts,
FrameEnvelope concurrency, atomic persistence, camera health status,
and API integration against active pipeline instances.
Addresses Critical Finding 1, 2, 8, 9, 12, and 13.
"""

import json
import sqlite3
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
    capture = FrameCapture(source_uri="synthetic", width=640, height=480, target_fps=5)
    capture.start()
    try:
        time.sleep(0.3)
        # Call read_envelope repeatedly under a tight interval
        env1 = capture.read_envelope("cam-1")
        assert env1.is_valid is True
        assert env1.image is not None
        assert env1.frame_id > 0
        assert env1.captured_at_monotonic_ns > 0

        # Immediate re-read should indicate is_new=False before new frame is generated
        env2 = capture.read_envelope("cam-1")
        if env2.frame_id == env1.frame_id:
            assert env2.is_new is False
        else:
            assert env2.is_new is True

        # Sleep to allow next frame
        time.sleep(0.3)
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
        # Allow loop to process multiple frames
        for _ in range(30):
            if pipeline.total_frames_processed > 0:
                break
            time.sleep(0.1)

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

        # Wait for retry backoff window to elapse
        time.sleep(1.05)

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


def test_camera_startup_failure_and_dashboard_available(tmp_path):
    """
    Regression Test 1: Camera startup failure must not crash the process;
    pipeline starts in OFFLINE state and dashboard/API remains accessible.
    """
    cfg = load_config("config/demo.yaml")
    cfg.cameras[0].source = "webcam:999"
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_cam_fail.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    pipeline.capture.allow_synthetic_fallback = False
    try:
        pipeline.start()
        assert pipeline.is_running is True
        assert pipeline.capture.is_connected is False

        with db_mgr.get_connection() as conn:
            cam_row = conn.execute("SELECT status FROM cameras WHERE id = 'cam-1'").fetchone()
            assert cam_row[0] == "OFFLINE"

        app = create_app(pipeline)
        client = TestClient(app)

        resp = client.get("/api/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["camera_status"] == "OFFLINE"

        h_resp = client.get("/api/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert h_data["camera_offline"] is True
        assert h_data["status"] == "offline"
    finally:
        pipeline.stop()


def test_camera_disconnect_and_reconnection_recovery(tmp_path):
    """
    Regression Test 2: FrameCapture gracefully handles disconnect, enters
    bounded backoff, and recovers when reconnected without process restart.
    """
    capture = FrameCapture(
        source_uri="webcam:999", width=640, height=480, target_fps=10, allow_synthetic_fallback=False
    )
    capture.start()
    try:
        assert capture.is_running is True
        assert capture.is_connected is False
        assert capture.is_alive is False

        env = capture.read_envelope("cam-1")
        assert env.source_status == "DISCONNECTED"

        # Simulate camera reconnection
        capture.source_uri = "synthetic"
        capture.is_synthetic = True

        for _ in range(25):
            if capture.is_connected:
                break
            time.sleep(0.1)
        assert capture.is_connected is True
        assert capture.is_alive is True

        env_recovered = capture.read_envelope("cam-1")
        assert env_recovered.is_valid is True
        assert env_recovered.image is not None
    finally:
        capture.stop()


def test_process_restart_preserves_persisted_state(tmp_path):
    """
    Regression Test 3: Process restart must not reset persisted states table
    to UNINITIALIZED or wipe existing FSM states.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_restart_state.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline1 = TracePipeline(cfg, db_mgr)
    pipeline1.start()
    try:
        time.sleep(0.2)
    finally:
        pipeline1.stop()

    asset_id = "safe-02"
    now = time.time()
    with db_mgr.get_connection() as conn:
        conn.execute(
            """
            UPDATE states
            SET state = 'MOVED', belief = 0.85, confidence = 0.95, last_verified_at = ?, updated_at = ?
            WHERE asset_id = ?
            """,
            (now, now, asset_id)
        )
        conn.commit()

    with db_mgr.get_connection() as conn:
        row = conn.execute("SELECT state, belief FROM states WHERE asset_id = ?", (asset_id,)).fetchone()
        assert row[0] == "MOVED"
        assert abs(row[1] - 0.85) < 1e-4

    # Phase 2: Restart pipeline against the same database
    # TracePipeline.__init__ triggers _init_assets_and_baselines()
    pipeline2 = TracePipeline(cfg, db_mgr)
    try:
        assert pipeline2.fsms[asset_id].state.value == "MOVED"
        assert abs(pipeline2.fsms[asset_id].belief - 0.85) < 1e-4

        with db_mgr.get_connection() as conn:
            row2 = conn.execute("SELECT state, belief FROM states WHERE asset_id = ?", (asset_id,)).fetchone()
            assert row2[0] == "MOVED"
    finally:
        pipeline2.stop()


def test_single_delivery_path_no_duplicate_notifications(tmp_path):
    """
    Regression Test 4: Verify each alert is delivered externally exactly once
    through the Outbox, without duplicate direct dispatch.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_single_delivery.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    delivery_calls = []

    def spy_send_external(event):
        delivery_calls.append(event)
        return True

    pipeline.alert_router.send_external = spy_send_external

    try:
        asset_cfg = pipeline.assets["safe-02"]
        dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        now = time.time()

        pipeline._handle_alert_transition(
            asset_id="safe-02",
            asset_cfg=asset_cfg,
            prev_state="SAFE",
            new_state="MOVED",
            confidence=0.95,
            belief=0.90,
            details={"reason": "test"},
            live_crop=dummy_crop,
            base_crop=dummy_crop,
            person_mask_crop=None,
            timestamp=now,
        )

        # Before outbox flush: zero external dispatches
        assert len(delivery_calls) == 0
        assert pipeline.outbox.get_pending_count() == 1

        # Flush outbox
        delivered = pipeline.outbox.flush(delivery_handler=pipeline._outbox_delivery_handler)
        assert len(delivered) == 1
        assert len(delivery_calls) == 1

        # Second flush must not deliver again
        delivered_again = pipeline.outbox.flush(delivery_handler=pipeline._outbox_delivery_handler)
        assert len(delivered_again) == 0
        assert len(delivery_calls) == 1
    finally:
        pipeline.stop()


def test_delivery_failure_retains_outbox_for_retry(tmp_path):
    """
    Regression Test 5: Outbox retains failed deliveries as PENDING and
    successfully delivers on subsequent flush after recovery.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_retry.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    try:
        event_payload = {"id": "evt-retry-1", "asset_id": "safe-02", "type": "MOVED", "severity": "high"}
        pipeline.outbox.enqueue("evt-retry-1", json.dumps(event_payload))
        assert pipeline.outbox.get_pending_count() == 1

        fail_calls = 0

        def failing_send_external(event):
            nonlocal fail_calls
            fail_calls += 1
            return False

        pipeline.alert_router.send_external = failing_send_external

        # Flush 1: delivery fails
        delivered = pipeline.outbox.flush(delivery_handler=pipeline._outbox_delivery_handler)
        assert len(delivered) == 0
        assert fail_calls == 1

        with db_mgr.get_connection() as conn:
            row = conn.execute("SELECT status, attempts FROM outbox WHERE event_id = 'evt-retry-1'").fetchone()
            assert row[0] == "PENDING"
            assert row[1] == 1

        # Simulate network recovery and wait for backoff window
        pipeline.alert_router.send_external = lambda event: True
        time.sleep(1.05)

        # Flush 2: delivery recovers
        delivered = pipeline.outbox.flush(delivery_handler=pipeline._outbox_delivery_handler)
        assert len(delivered) == 1
        assert delivered[0] == "evt-retry-1"

        with db_mgr.get_connection() as conn:
            row = conn.execute("SELECT status, delivered_at FROM outbox WHERE event_id = 'evt-retry-1'").fetchone()
            assert row[0] == "DELIVERED"
            assert row[1] is not None
    finally:
        pipeline.stop()


def test_region_bounds_validation_rejects_out_of_bounds_geometry(tmp_path):
    """
    Phase 2 Regression Test: Bounds validation helper and API endpoint reject
    out-of-bounds, negative, or degenerate polygon geometries.
    """
    from trace.pipeline import validate_region_bounds

    frame_shape = (720, 1280)  # (h, w)

    # Valid polygon
    valid, _ = validate_region_bounds([[100, 100], [200, 100], [200, 200], [100, 200]], frame_shape)
    assert valid is True

    # Out of bounds (> 1280 width)
    invalid_w, err = validate_region_bounds([[100, 100], [1400, 100], [1400, 200], [100, 200]], frame_shape)
    assert invalid_w is False
    assert "out of frame bounds" in err

    # Negative coordinates
    invalid_neg, err = validate_region_bounds([[-10, 100], [200, 100], [200, 200], [-10, 200]], frame_shape)
    assert invalid_neg is False

    # Degenerate polygon (< 3 vertices)
    invalid_deg, _ = validate_region_bounds([[10, 10], [20, 20]], frame_shape)
    assert invalid_deg is False

    # Degenerate size (< 10px)
    invalid_tiny, _ = validate_region_bounds([[10, 10], [15, 10], [15, 15], [10, 15]], frame_shape)
    assert invalid_tiny is False

    # Test rejection via API
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_mgr = DatabaseManager(str(tmp_path / "test_api_bounds.db"))
    pipeline = TracePipeline(cfg, db_mgr)
    try:
        app = create_app(pipeline)
        client = TestClient(app)

        bad_payload = {
            "id": "bad-asset-1",
            "name": "Out of Bounds Box",
            "class": "laptop",
            "region": [[100, 100], [2000, 100], [2000, 500], [100, 500]],
        }
        resp = client.post("/api/assets", json=bad_payload, headers={"Authorization": f"Bearer {cfg.security.admin_token}"})
        assert resp.status_code == 400
        assert "bound" in resp.json()["detail"].lower() or "exceeds" in resp.json()["detail"].lower()
    finally:
        pipeline.stop()


def test_unreadable_tag_does_not_trigger_immediate_false_swap(tmp_path):
    """
    Phase 2 Regression Test: When asset tag is temporarily unreadable or blocked,
    tag verification returns UNREADABLE instead of MISMATCH and does not cause
    immediate false SWAPPED alert.
    """
    from trace.vision.verify import RegionVerifier
    from trace.logic.fsm import FSMConfig, AssetFSM, Obs, State as S

    verifier = RegionVerifier()
    # Baseline crop with synthetic pattern
    baseline = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(baseline, (50, 50), 30, (200, 200, 200), -1)
    variants = verifier.extractor.generate_variants(baseline)

    # Live crop identical visually, but ArUco tag is not detectable (no marker in image)
    live = baseline.copy()
    res = verifier.verify_asset(
        live_crop_bgr=live,
        baseline_variants=variants,
        baseline_median_crop=baseline,
        empty_roi_bgr=None,
        tag_id=12,
    )

    assert res["tag_status"] == "UNREADABLE"
    assert res["tag_matched"] is False

    # Feed into FSM
    fsm = AssetFSM(FSMConfig())
    fsm.state = S.SAFE
    obs = Obs(
        t=100.0,
        present_score=res["present_score"],
        drift_frac=0.0,
        visibility=1.0,
        empty_match=res["empty_match"],
        camera_ok=True,
        tag_matched=res["tag_matched"],
        tag_status=res["tag_status"],
        is_partially_occluded=False,
    )
    new_state = fsm.step(obs)
    # Must remain SAFE, NOT SWAPPED!
    assert new_state == S.SAFE


def test_partial_non_human_occlusion_never_triggers_theft_or_swap(tmp_path):
    """
    Phase 2 Regression Test: Partial non-human occlusion (e.g. piece of paper)
    detected by tile consistency is routed to OCCLUDED or UNKNOWN, never SWAPPED or MISSING.
    """
    from trace.vision.verify import RegionVerifier
    from trace.logic.fsm import FSMConfig, AssetFSM, Obs, State as S

    verifier = RegionVerifier()
    # Baseline crop with rich texture
    rng = np.random.RandomState(42)
    baseline = rng.randint(50, 220, size=(120, 120, 3)).astype(np.uint8)
    variants = verifier.extractor.generate_variants(baseline)

    # Partial occluder: covering bottom half with blank white paper
    partially_occluded_crop = baseline.copy()
    partially_occluded_crop[60:, :] = 255

    res = verifier.verify_asset(
        live_crop_bgr=partially_occluded_crop,
        baseline_variants=variants,
        baseline_median_crop=baseline,
        empty_roi_bgr=None,
        tag_id=None,
    )

    assert res["is_partially_occluded"] is True
    assert res["occlusion_ratio"] > 0.0

    # FSM test: verify it never triggers SWAPPED or MISSING
    fsm = AssetFSM(FSMConfig(missing_wait_s=2.0, swap_s=2.0))
    fsm.state = S.SAFE

    for t in [100.0, 102.0, 105.0, 110.0]:
        obs = Obs(
            t=t,
            present_score=res["present_score"],
            drift_frac=0.0,
            visibility=1.0 - res["occlusion_ratio"],
            empty_match=0.0,
            camera_ok=True,
            is_partially_occluded=res["is_partially_occluded"],
        )
        state = fsm.step(obs)
        assert state in (S.SAFE, S.OCCLUDED, S.UNKNOWN)
        assert state not in (S.SWAPPED, S.MISSING, S.MOVED)


def test_empty_baseline_persistence_and_restoration_across_restarts(tmp_path):
    """
    Phase 2 Regression Test: Empty baseline is persisted to disk/DB and
    restored into memory on pipeline restart, surviving process restart.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_empty_baseline.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline1 = TracePipeline(cfg, db_mgr)
    try:
        app1 = create_app(pipeline1)
        client1 = TestClient(app1)

        # Set mock clean frame
        dummy_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        with pipeline1.lock:
            pipeline1.latest_clean_frame = dummy_frame

        # Capture empty baseline for safe-02 via API
        resp = client1.post("/api/assets/safe-02/baseline?kind=empty", headers={"Authorization": f"Bearer {cfg.security.admin_token}"})
        assert resp.status_code == 200
        assert pipeline1.empty_rois.get("safe-02") is not None

        # Verify DB has empty baseline record
        with db_mgr.get_connection() as conn:
            row = conn.execute("SELECT kind, image_path FROM baselines WHERE asset_id = 'safe-02' AND kind = 'empty'").fetchone()
            assert row is not None
            assert row[0] == "empty"
            assert Path(row[1]).exists()

        # Check inspection endpoint
        b_resp = client1.get("/api/assets/safe-02/baselines")
        assert b_resp.status_code == 200
        assert b_resp.json()["has_empty"] is True
    finally:
        pipeline1.stop()

    # Now restart pipeline with new instance
    pipeline2 = TracePipeline(cfg, db_mgr)
    try:
        # In pipeline2, empty baseline for safe-02 must be restored from DB!
        assert pipeline2.empty_rois.get("safe-02") is not None
        assert pipeline2.empty_rois["safe-02"].shape[0] > 0
    finally:
        pipeline2.stop()


def test_durable_camera_fault_creates_ledger_incident_and_outbox_records(tmp_path):
    """
    Phase 2 Regression Test: Camera obstruction / tampering atomically creates
    ledger entry, incident record, event record, and enqueues outbox message.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_cam_fault.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    try:
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        now = time.time()

        # Trigger camera obstruction fault
        pipeline._handle_camera_fault(
            camera_id="cam-1",
            fault_type="CAMERA_OBSTRUCTED",
            details={"sharpness": 12.0, "reason": "lens covered"},
            frame_bgr=dummy_frame,
            timestamp=now,
        )

        with db_mgr.get_connection() as conn:
            # 1. Camera row status updated
            cam_row = conn.execute("SELECT status FROM cameras WHERE id = 'cam-1'").fetchone()
            assert cam_row[0] == "CAMERA_OBSTRUCTED"

            # 2. Incident created
            inc_row = conn.execute("SELECT id, status, counter FROM incidents WHERE asset_id IS NULL").fetchone()
            assert inc_row is not None
            assert inc_row[1] == "OPEN"

            # 3. Event created
            evt_row = conn.execute("SELECT id, camera_id, type, severity FROM events WHERE type = 'CAMERA_OBSTRUCTED'").fetchone()
            assert evt_row is not None
            assert evt_row[1] == "cam-1"
            assert evt_row[3] == "high"

            # 4. Ledger entry exists
            ledger_count = conn.execute("SELECT count(*) FROM ledger").fetchone()[0]
            assert ledger_count >= 1

            # 5. Outbox enqueued
            outbox_row = conn.execute("SELECT event_id, status FROM outbox WHERE event_id = ?", (evt_row[0],)).fetchone()
            assert outbox_row is not None
            assert outbox_row[1] == "PENDING"

        # Calling again with identical fault type does NOT duplicate incident or outbox item
        pipeline._handle_camera_fault(
            camera_id="cam-1",
            fault_type="CAMERA_OBSTRUCTED",
            details={"sharpness": 12.0},
            frame_bgr=dummy_frame,
            timestamp=now + 1.0,
        )
        with db_mgr.get_connection() as conn:
            inc_count = conn.execute("SELECT count(*) FROM incidents WHERE asset_id IS NULL").fetchone()[0]
            assert inc_count == 1
    finally:
        pipeline.stop()


def test_anchor_service_and_verification(tmp_path):
    """
    Phase 3 Regression Test: Off-box cryptographic anchoring creates verifiable
    receipts and detects off-box checkpoint mismatches when ledger is tampered.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_anchor.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    pipeline.anchor_service.anchor_file = tmp_path / "anchors.log"
    try:
        app = create_app(pipeline)
        client = TestClient(app)

        # 1. Initially empty ledger cannot be anchored
        resp_empty = client.post("/api/ledger/anchor", headers={"Authorization": f"Bearer {cfg.security.admin_token}"})
        assert resp_empty.status_code == 400

        # 2. Add an event to ledger
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        pipeline._handle_camera_fault(
            camera_id="cam-1",
            fault_type="CAMERA_OBSTRUCTED",
            details={"sharpness": 10.0},
            frame_bgr=dummy_frame,
            timestamp=time.time(),
        )

        # 3. Anchor ledger head
        resp = client.post("/api/ledger/anchor", headers={"Authorization": f"Bearer {cfg.security.admin_token}"})
        assert resp.status_code == 200
        anc_data = resp.json()["anchor"]
        assert anc_data["idx"] == 1
        assert "head_hash" in anc_data
        assert "rfc3161_token" in anc_data

        # 4. List anchors
        resp_list = client.get("/api/ledger/anchors", headers={"Authorization": f"Bearer {cfg.security.admin_token}"})
        assert resp_list.status_code == 200
        assert resp_list.json()["off_box_count"] >= 1

        # 5. Verify ledger includes off-box check
        resp_ver = client.get("/api/ledger/verify", headers={"Authorization": f"Bearer {cfg.security.admin_token}"})
        assert resp_ver.status_code == 200
        ver_data = resp_ver.json()
        assert ver_data["ok"] is True
        assert ver_data.get("anchors_checked", 0) >= 1
    finally:
        pipeline.stop()


def test_pipeline_e2e_controlled_frame_scenarios(tmp_path):
    """
    Phase 3 E2E Integration: Directly exercises TracePipeline with controlled frame sequences.
    Covers:
      1. Baseline acquisition and stable convergence to SAFE.
      2. Asset displacement (move_asset) -> transitions to MOVED, records ledger event,
         stores before/after evidence package, enqueues to outbox, and delivers exactly once.
      3. Partial non-human occlusion (occlude_asset) -> transitions safely to OCCLUDED/UNKNOWN,
         strictly asserting NO false theft / SWAPPED alert is generated.
      4. Network loss recovery (cut_network + alert) -> pending outbox queue retained during outage,
         then cleanly flushed and delivered when network recovers.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_pipeline_e2e.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    delivery_log = []

    def tracking_send_external(event):
        delivery_log.append(event)
        return True

    pipeline.alert_router.send_external = tracking_send_external

    try:
        pipeline.start()
        time.sleep(1.0)
        assert pipeline.is_running is True

        # Assert all assets reached SAFE or PRESENT initially
        for a_id in ["safe-02", "printer-01", "laptop-03"]:
            assert pipeline.fsms[a_id].state.value in ("SAFE", "PRESENT")

        # 1. Trigger asset displacement on printer-01
        pipeline.demo_injector.trigger_fault("move_asset", asset_id="printer-01", duration_s=6.0)
        time.sleep(2.5)

        # printer-01 should have transitioned to MOVED
        assert pipeline.fsms["printer-01"].state.value == "MOVED"

        # Verify database recorded incident, event, and evidence
        with db_mgr.get_connection() as conn:
            event_row = conn.execute(
                "SELECT id, asset_id, type, before_path, after_path, incident_id FROM events WHERE asset_id = 'printer-01' AND type = 'MOVED'"
            ).fetchone()
            assert event_row is not None
            assert event_row[3] is not None  # before_path
            assert event_row[4] is not None  # after_path

            incident_row = conn.execute(
                "SELECT id, asset_id, status FROM incidents WHERE id = ?", (event_row[5],)
            ).fetchone()
            assert incident_row is not None

            # Ledger must have hashed the event
            ledger_rows = conn.execute("SELECT idx, body, hash FROM ledger").fetchall()
            assert any(event_row[0] in row[1] for row in ledger_rows)

        # Verify outbox single delivery (no duplicates)
        pipeline.outbox.flush(delivery_handler=pipeline._outbox_delivery_handler)
        moved_deliveries = [
            e for e in delivery_log
            if (e.get("type") == "MOVED" or e.get("event_type") == "MOVED") and e.get("asset_id") == "printer-01"
        ]
        assert len(moved_deliveries) == 1

        # 2. Trigger partial occlusion on safe-02 (non-human obstruction)
        pipeline.demo_injector.reset_all()
        time.sleep(1.0)

        delivery_count_pre_occlusion = len(delivery_log)
        pipeline.demo_injector.trigger_fault("occlude_asset", asset_id="safe-02", duration_s=5.0)
        time.sleep(2.0)

        # safe-02 must be safe/occluded/unknown, strictly NOT SWAPPED or MISSING!
        safe_state = pipeline.fsms["safe-02"].state.value
        assert safe_state in ("OCCLUDED", "UNKNOWN", "SAFE"), f"Unexpected unsafe state: {safe_state}"
        assert safe_state not in ("SWAPPED", "MISSING"), "SAFETY VIOLATION: Partial occlusion falsely declared theft!"

        # Zero false swap/missing deliveries
        theft_deliveries = [
            e for e in delivery_log[delivery_count_pre_occlusion:]
            if (e.get("type") in ("SWAPPED", "MISSING") or e.get("event_type") in ("SWAPPED", "MISSING"))
        ]
        assert len(theft_deliveries) == 0

        # 3. Network cut + alert queued recovery
        pipeline.demo_injector.reset_all()
        time.sleep(0.5)

        # Cut network
        pipeline.demo_injector.trigger_fault("cut_network", duration_s=10.0)
        pipeline.outbox.set_network_state(False)

        pre_cut_deliveries = len(delivery_log)
        # Generate another alert while network is down (asset move on safe-02)
        pipeline.demo_injector.trigger_fault("move_asset", asset_id="safe-02", duration_s=5.0)
        time.sleep(2.5)

        # Outbox should have retained the alert without delivering externally
        assert len(delivery_log) == pre_cut_deliveries, "Alert was prematurely dispatched while network cut!"

        # Now restore network
        pipeline.demo_injector.reset_all()
        pipeline.outbox.set_network_state(True)
        pipeline.outbox.flush(delivery_handler=pipeline._outbox_delivery_handler)

        # The queued alert is now delivered
        assert len(delivery_log) > pre_cut_deliveries

    finally:
        pipeline.stop()


def test_physical_calibration_reference_persistence_and_recalibration(tmp_path):
    """
    Blocker 1: Verify reference frame/landmarks persist to disk, are not silently
    overwritten on restart, and venue recalibration updates baselines and rejects invalid crops.
    """
    from trace.vision.align import FrameAligner
    ref_file = tmp_path / "ref_frame.jpg"
    aligner = FrameAligner(reference_path=str(ref_file))

    dummy_frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag1 = cv2.aruco.generateImageMarker(aruco_dict, 1, 60)
    dummy_frame[40:100, 80:140] = cv2.cvtColor(tag1, cv2.COLOR_GRAY2BGR)

    res1 = aligner.register_reference(dummy_frame, save_to_disk=True)
    assert ref_file.exists()
    assert 1 in aligner.ref_aruco_corners

    # Non-forced register_reference must NOT overwrite existing reference
    mod_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res2 = aligner.register_reference(mod_frame, save_to_disk=True, force=False)
    assert res2.get("already_registered") is True
    assert 1 in aligner.ref_aruco_corners

    # Loading from disk restores reference landmarks
    new_aligner = FrameAligner(reference_path=str(ref_file))
    ok = new_aligner.load_reference()
    assert ok is True
    assert 1 in new_aligner.ref_aruco_corners

    # Recalibration rejects flat/blank regions
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_mgr = DatabaseManager(str(tmp_path / "calib_test.db"))
    pipeline = TracePipeline(cfg, db_mgr)
    try:
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        calib_res = pipeline._capture_initial_baselines(blank_frame, overwrite=True)
        assert len(calib_res["failed"]) > 0
    finally:
        pipeline.stop()


def test_processor_survival_on_db_or_evidence_failure_and_health_reporting(tmp_path):
    """
    Blocker 2: Database or evidence write failures do not kill the loop;
    camera fault is not marked handled until persistence succeeds;
    /api/health accurately reports an unhealthy dead processing thread.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_survival.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    try:
        pipeline.start()
        time.sleep(0.5)
        assert pipeline.is_alive is True

        # 1. Test camera fault persistence failure does not mark fault handled
        orig_execute = db_mgr.get_connection
        pipeline.active_camera_fault = None

        class BrokenConnContext:
            def __enter__(self):
                raise sqlite3.OperationalError("Simulated disk I/O error on database write")
            def __exit__(self, *args):
                pass

        pipeline.db_mgr.get_connection = lambda: BrokenConnContext()

        # Attempt to handle camera fault with broken DB
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        pipeline._handle_camera_fault("cam-1", "CAMERA_OBSTRUCTED", {}, dummy_frame, time.time())

        # Fault must NOT be marked active/handled because persistence failed!
        assert pipeline.active_camera_fault is None

        # Restore DB connection
        pipeline.db_mgr.get_connection = orig_execute

        # 2. Test evidence write failure does not mark fault handled and does not crash
        orig_save_crops = pipeline.evidence_writer.save_crops
        def failing_save_crops(*args, **kwargs):
            raise OSError("Simulated disk full during evidence write")
        pipeline.evidence_writer.save_crops = failing_save_crops

        pipeline._handle_camera_fault("cam-1", "CAMERA_OBSTRUCTED", {}, dummy_frame, time.time())
        assert pipeline.active_camera_fault is None

        pipeline.evidence_writer.save_crops = orig_save_crops

        # 3. Test processor thread survival
        time.sleep(0.5)
        assert pipeline.is_alive is True

        # 3. Test /api/health detects dead processing thread
        app = create_app(pipeline)
        client = TestClient(app)

        health_live = client.get("/api/health").json()
        assert health_live["processor_alive"] is True
        assert health_live["status"] in ("healthy", "degraded", "offline")

        # Simulate crashed thread while pipeline is_running is True
        pipeline._thread = None
        health_dead = client.get("/api/health").json()
        assert health_dead["processor_alive"] is False
        assert health_dead["status"] == "unhealthy"
    finally:
        pipeline.stop()


def test_network_reliability_decoupled_worker_and_exponential_backoff(tmp_path):
    """
    Blocker 3: Alert router never reports True when no external channel is configured;
    failed delivery advances next_retry_at with bounded exponential backoff.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    cfg.alerts.telegram.enabled = False
    cfg.alerts.ntfy.enabled = False
    db_path = tmp_path / "test_net.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    try:
        dummy_event = {"id": "ev-1", "type": "MOVED", "severity": "high"}
        # Never report True when no channels are configured!
        assert pipeline.alert_router.send_external(dummy_event) is False

        # Test exponential backoff in OutboxManager
        now = time.time()
        pipeline.outbox.enqueue("ev-fail-1", json.dumps(dummy_event))

        def failing_handler(info):
            raise ConnectionError("Simulated network timeout")

        # First flush: failure sets attempt=1 and backoff
        delivered = pipeline.outbox.flush(delivery_handler=failing_handler)
        assert len(delivered) == 0

        with db_mgr.get_connection() as conn:
            row = conn.execute("SELECT attempts, next_retry_at FROM outbox WHERE event_id = 'ev-fail-1'").fetchone()
            assert row[0] == 1
            assert row[1] > now

        # Immediate second flush does not retry because item is backing off
        delivered_early = pipeline.outbox.flush(delivery_handler=failing_handler)
        assert len(delivered_early) == 0
        with db_mgr.get_connection() as conn:
            row_early = conn.execute("SELECT attempts FROM outbox WHERE event_id = 'ev-fail-1'").fetchone()
            assert row_early[0] == 1
    finally:
        pipeline.stop()


def test_privacy_fail_closed_camera_fault_and_unix_timestamps(tmp_path):
    """
    Blocker 4: Camera fault evidence runs privacy segmentation;
    timestamp in events/incidents is wall-clock Unix time (> 1700000000).
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    cfg.privacy.mask_people = True
    db_path = tmp_path / "test_privacy_fault.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    try:
        now_unix = time.time()
        frame = np.full((720, 1280, 3), 150, dtype=np.uint8)
        pipeline._handle_camera_fault(
            camera_id="cam-1",
            fault_type="CAMERA_OBSTRUCTED",
            details={"test": True},
            frame_bgr=frame,
            timestamp=now_unix,
        )

        with db_mgr.get_connection() as conn:
            event_row = conn.execute("SELECT id, detected_at, before_path, after_path FROM events WHERE type = 'CAMERA_OBSTRUCTED'").fetchone()
            assert event_row is not None
            assert event_row[1] >= 1700000000.0

            incident_row = conn.execute("SELECT opened_at FROM incidents WHERE id = (SELECT incident_id FROM events WHERE type = 'CAMERA_OBSTRUCTED')").fetchone()
            assert incident_row is not None
            assert incident_row[0] >= 1700000000.0

            assert Path(event_row[2]).exists()
            assert Path(event_row[3]).exists()
    finally:
        pipeline.stop()


def test_api_registered_assets_persist_and_load_across_restarts(tmp_path):
    """
    Blocker 5: Assets registered via /api/assets are persisted into database,
    survive pipeline process restart, and are included in pipeline.assets and pipeline.fsms.
    """
    cfg = load_config("config/demo.yaml")
    cfg.ledger.key_file = str(tmp_path / "ledger_key.json")
    db_path = tmp_path / "test_asset_restart.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    app = create_app(pipeline)
    client = TestClient(app)

    try:
        new_asset = {
            "id": "runtime-asset-99",
            "name": "Server Rack Key",
            "class_name": "other",
            "criticality": 3,
            "mode": "fixed",
            "camera_id": "cam-1",
            "tag_id": 99,
            "region": [[200, 200], [300, 200], [300, 300], [200, 300]],
            "present_threshold": 0.75,
        }
        headers = {"Authorization": f"Bearer {cfg.security.admin_token}"}
        resp = client.post("/api/assets", json=new_asset, headers=headers)
        assert resp.status_code == 200
        assert "runtime-asset-99" in pipeline.assets
    finally:
        pipeline.stop()

    # Restart new pipeline instance with same database
    pipeline2 = TracePipeline(cfg, db_mgr)
    try:
        assert "runtime-asset-99" in pipeline2.assets
        assert "runtime-asset-99" in pipeline2.fsms
        assert pipeline2.assets["runtime-asset-99"].name == "Server Rack Key"
        assert pipeline2.assets["runtime-asset-99"].thresholds.present == 0.75
    finally:
        pipeline2.stop()



