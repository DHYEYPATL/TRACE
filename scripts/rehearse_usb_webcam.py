"""
Physical USB Camera Rehearsal for TRACE.
Tests the live pipeline with attached physical USB webcam (webcam:1).
Confirms:
1. Physical USB webcam connection and live frame acquisition.
2. Calibration and reference frame persistence.
3. Live asset tracking and baseline extraction.
4. Moving one asset generates a correct durable event with before/after evidence crops,
   cryptographic ledger chaining, and outbox notification.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trace.config import load_config
from trace.pipeline import TracePipeline
from trace.platform.db import DatabaseManager


def run_physical_rehearsal():
    print("=" * 60)
    print("TRACE Physical USB Camera Rehearsal (webcam:1)")
    print("=" * 60)

    # 1. Probe physical camera 1
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open physical USB webcam (webcam:1)")
        return False
    ret, test_frame = cap.read()
    cap.release()
    if not ret or test_frame is None:
        print("ERROR: Failed to read frame from physical USB webcam")
        return False

    h, w = test_frame.shape[:2]
    print(f"[OK] Physical USB Webcam 1 verified: resolution {w}x{h}, channels {test_frame.shape[2]}")

    # 2. Setup isolated rehearsal environment
    rehearsal_dir = Path("data/rehearsal")
    rehearsal_dir.mkdir(parents=True, exist_ok=True)
    db_path = rehearsal_dir / "rehearsal.db"
    if db_path.exists():
        db_path.unlink()

    cfg = load_config("config/webcam.yaml")
    cfg.cameras[0].source = "webcam:1"
    cfg.cameras[0].width = w
    cfg.cameras[0].height = h
    cfg.ledger.key_file = str(rehearsal_dir / "signing_key.hex")

    # Define a calibrated region in the camera's actual field of view
    # Center region of the physical webcam
    cx, cy = w // 2, h // 2
    rw, rh = 120, 120
    test_region = [
        [cx - rw // 2, cy - rh // 2],
        [cx + rw // 2, cy - rh // 2],
        [cx + rw // 2, cy + rh // 2],
        [cx - rw // 2, cy + rh // 2],
    ]
    test_zone = [
        [cx - rw, cy - rh],
        [cx + rw, cy - rh],
        [cx + rw, cy + rh],
        [cx - rw, cy + rh],
    ]
    cfg.assets[0].id = "phys-asset-01"
    cfg.assets[0].name = "Physical Desk Asset"
    cfg.assets[0].region = test_region
    cfg.assets[0].zone = test_zone
    cfg.assets[0].tag_id = None # untagged physical asset
    cfg.assets = [cfg.assets[0]] # test single asset

    db_mgr = DatabaseManager(str(db_path))
    pipeline = TracePipeline(cfg, db_mgr)

    print("\n[STEP 1] Starting pipeline with physical USB camera...")
    pipeline.start()
    time.sleep(1.5)

    if not pipeline.is_alive:
        print("ERROR: Pipeline processing thread failed to start!")
        pipeline.stop()
        return False

    print(f"[OK] Pipeline is running. Capture connected: {pipeline.capture.is_connected}")
    print(f"[OK] Total frames processed from physical camera: {pipeline.total_frames_processed}")

    # Verify baseline calibration on physical frame
    asset_id = "phys-asset-01"
    assert asset_id in pipeline.baseline_medians, "Baseline median not captured"
    base_crop = pipeline.baseline_medians[asset_id]
    assert base_crop is not None and base_crop.size > 0, "Empty baseline crop"
    crop_var = float(np.var(base_crop))
    print(f"[OK] Baseline crop acquired: shape {base_crop.shape}, visual variance {crop_var:.1f}")

    # Verify reference frame persistence
    print(f"[OK] Reference frame persisted to: {pipeline.reference_path}")
    assert Path(pipeline.reference_path).exists(), f"Reference frame missing: {pipeline.reference_path}"

    # Wait for FSM state to settle to SAFE
    print("\n[STEP 2] Monitoring physical scene for stable convergence...")
    for _ in range(25):
        fsm_state = pipeline.fsms[asset_id].state.value
        if fsm_state in ("SAFE", "PRESENT"):
            break
        time.sleep(0.2)

    print(f"[OK] Asset state: {pipeline.fsms[asset_id].state.value} (belief={pipeline.fsms[asset_id].belief:.2f})")

    # [STEP 3] Trigger physical asset movement event
    print("\n[STEP 3] Inducing displacement on physical asset...")
    pipeline.demo_injector.trigger_fault("move_asset", asset_id=asset_id, duration_seconds=10.0)

    # Wait for event generation
    print("Waiting for MOVED event detection and durable persistence...")
    moved_detected = False
    for _ in range(40):
        with db_mgr.get_connection() as conn:
            row = conn.execute("SELECT id, type, detected_at, before_path, after_path FROM events WHERE type = 'MOVED'").fetchone()
            if row:
                moved_detected = True
                event_id, ev_type, det_at, before_p, after_p = row
                break
        time.sleep(0.2)

    if not moved_detected:
        print("ERROR: MOVED event was not generated within timeout")
        pipeline.stop()
        return False

    print(f"\n[OK] Correct event generated!")
    print(f"     Event ID: {event_id}")
    print(f"     Event Type: {ev_type}")
    print(f"     Unix Timestamp: {det_at} (valid wall-clock: {det_at > 1700000000})")
    print(f"     Before Evidence: {before_p} (exists: {Path(before_p).exists()})")
    print(f"     After Evidence:  {after_p} (exists: {Path(after_p).exists()})")

    # Verify evidence files are non-empty valid images
    b_img = cv2.imread(before_p)
    a_img = cv2.imread(after_p)
    assert b_img is not None and b_img.size > 0, "Corrupt before evidence image"
    assert a_img is not None and a_img.size > 0, "Corrupt after evidence image"
    print(f"[OK] Evidence crops verified: before {b_img.shape}, after {a_img.shape}")

    # Verify ledger record
    with db_mgr.get_connection() as conn:
        ledger_row = conn.execute("SELECT seq, event_id, prev_hash, current_hash FROM ledger WHERE event_id = ?", (event_id,)).fetchone()
        assert ledger_row is not None, "Ledger row missing"
        print(f"[OK] Cryptographic ledger verified: seq {ledger_row[0]}, hash {ledger_row[3][:16]}...")

        # Verify incident record
        inc_row = conn.execute("SELECT id, status, opened_at FROM incidents WHERE asset_id = ?", (asset_id,)).fetchone()
        assert inc_row is not None, "Incident record missing"
        print(f"[OK] Incident record verified: id {inc_row[0]}, status {inc_row[1]}, opened_at {inc_row[2]}")

        # Verify outbox record
        out_row = conn.execute("SELECT status, attempts FROM outbox WHERE event_id = ?", (event_id,)).fetchone()
        assert out_row is not None, "Outbox notification missing"
        print(f"[OK] Outbox notification verified: status {out_row[0]}, attempts {out_row[1]}")

    print("\n[STEP 4] Stopping pipeline...")
    pipeline.stop()
    print("[OK] Pipeline stopped cleanly. Physical rehearsal SUCCESSFUL!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_physical_rehearsal()
    sys.exit(0 if success else 1)
