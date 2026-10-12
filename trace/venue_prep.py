"""
Venue Preparation & Physical Setup Utility.
Generates printable ArUco tag stickers and executes 60-second venue lighting re-calibration (Section 9.3 & 12.4).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np


def generate_aruco_stickers(out_dir: str = "data/aruco_stickers") -> None:
    """
    Generates high-resolution printable PNG stickers for environmental landmarks and assets
    using OpenCV ArUco Dictionary 4x4_50 matching Section 5.1 & 5.3.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Markers list: (ID, filename, label)
    markers = [
        (1, "wall_landmark_1.png", "WALL LANDMARK #1 (Top-Left)"),
        (2, "wall_landmark_2.png", "WALL LANDMARK #2 (Top-Right)"),
        (3, "wall_landmark_3.png", "WALL LANDMARK #3 (Center)"),
        (12, "tag_safe_02.png", "ASSET TAG: Cash & Gold Safe (ID: 12)"),
        (24, "tag_printer_01.png", "ASSET TAG: Laser Printer (ID: 24)"),
        (31, "tag_laptop_03.png", "ASSET TAG: Laptop Workstation (ID: 31)"),
    ]

    for m_id, fname, label in markers:
        # Generate 400x400 high-res marker image
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, m_id, 400)
        # Pad with white border and human-readable label
        canvas = np.full((500, 440), 255, dtype=np.uint8)
        canvas[20:420, 20:420] = marker_img
        cv2.putText(canvas, label, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)

        cv2.imwrite(str(p / fname), canvas)
        print(f"  -> Generated printable marker: {fname} ({label})")

    print(f"\nAll printable stickers generated successfully in: {p.resolve()}")


def recalibrate_venue_lighting(config_path: str = "config/demo.yaml") -> None:
    """
    Snaps the live presentation camera feed and updates reference frame baselines
    to adapt to the venue's illumination 60 minutes before the pitch (Section 12.4).
    """
    import time
    from trace.config import load_config
    from trace.platform.db import get_db
    from trace.pipeline import TracePipeline

    print("Connecting to camera and capturing venue illumination reference...")
    cfg = load_config(config_path)
    db_mgr = get_db("data/trace.db")
    pipeline = TracePipeline(cfg, db_mgr)
    pipeline.start()

    time.sleep(1.0)
    success, frame, _ = pipeline.capture.read_latest()
    if success and frame is not None:
        res = pipeline._capture_initial_baselines(frame, overwrite=True)
        print("  -> Reference frame captured & persisted.")
        print(f"  -> Calibrated baselines for {len(res.get('calibrated', []))} assets: {res.get('calibrated', [])}")
        if res.get("failed"):
            print(f"  -> WARNING: Baseline calibration failed for {len(res['failed'])} assets:")
            for f in res["failed"]:
                print(f"     - {f['asset_id']}: {f['reason']}")
        print("Venue calibration complete!")
    else:
        print("Failed to capture camera frame. Check camera connection.")

    pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRACE Venue Preparation & ArUco Marker Generator")
    parser.add_argument("--generate-markers", action="store_true", help="Generate printable ArUco marker stickers")
    parser.add_argument("--recalibrate", action="store_true", help="Re-calibrate baselines for venue lighting")
    parser.add_argument("--config", type=str, default="config/webcam.yaml", help="Path to config file for calibration (default: config/webcam.yaml)")
    args = parser.parse_args()

    if args.recalibrate:
        recalibrate_venue_lighting(args.config)
    else:
        generate_aruco_stickers()
