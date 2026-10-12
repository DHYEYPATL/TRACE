"""
Camera Health Monitor module.
Evaluates sharpness, brightness, texture, landmark visibility, and frame freeze.
Implements CAMERA_OBSTRUCTED and SYSTEM_OFFLINE state reasoning per Section 5.2.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class CameraHealthMonitor:
    """
    Monitors optical health of the camera feed and detects obstructions, tampering, and outages.
    """

    def __init__(self, hold_sec: float = 1.5, release_sec: float = 3.0):
        self.hold_sec = hold_sec
        self.release_sec = release_sec

        # Rolling baseline estimates
        self.baseline_sharp: float = 180.0
        self.baseline_bright: float = 0.65
        self.baseline_tiles: List[float] = [0.05] * 16
        self.has_calibrated_baseline: bool = False

        # State tracking
        self.fail_start_time: Optional[float] = None
        self.pass_start_time: Optional[float] = None
        self.is_obstructed: bool = False
        self.is_offline: bool = False

        # Freeze check
        self.last_frame_hash: Optional[int] = None
        self.last_unique_frame_time: float = time.time()
        self.last_frame_received_time: float = time.time()

        # Rolling history of brightness (timestamp, value) for sudden drop detection
        self.bright_history = collections.deque(maxlen=30)

    def calibrate_baseline(self, frame_bgr: np.ndarray) -> None:
        """Sets the baseline reference values from clean reference frames."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.baseline_sharp = max(20.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        img_var = max(1.0, float(np.var(gray)))
        self.baseline_rel_sharp = self.baseline_sharp / img_var

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        self.baseline_bright = float(hsv[..., 2].mean() / 255.0)

        med = float(np.median(gray))
        low_th = max(10, int(0.66 * med))
        high_th = min(250, int(1.33 * med))
        edges = cv2.Canny(gray, low_th, high_th) > 0
        h, w = edges.shape
        self.baseline_tiles = [
            float(edges[i * h // 4 : (i + 1) * h // 4, j * w // 4 : (j + 1) * w // 4].mean())
            for i in range(4)
            for j in range(4)
        ]
        self.has_calibrated_baseline = True

    def check(
        self,
        frame_bgr: Optional[np.ndarray],
        timestamp: float,
        landmark_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluates frame metrics and returns health status.
        """
        now = timestamp
        # 1. Offline & Frozen frame check
        if frame_bgr is None:
            if now - self.last_frame_received_time > 3.0:
                self.is_offline = True
            return {
                "ok": False,
                "status": "SYSTEM_OFFLINE",
                "is_obstructed": False,
                "is_offline": True,
                "reason": "No frame received for > 3.0s"
            }

        self.last_frame_received_time = now

        # Quick hash for frozen check (downsample to 32x32 for speed)
        tiny = cv2.resize(frame_bgr, (32, 32), interpolation=cv2.INTER_NEAREST)
        f_hash = hash(tiny.tobytes())
        if f_hash != self.last_frame_hash:
            self.last_frame_hash = f_hash
            self.last_unique_frame_time = now
        elif now - self.last_unique_frame_time > 2.0:
            # Frozen frame for > 2 seconds
            self.is_offline = True
            return {
                "ok": False,
                "status": "SYSTEM_OFFLINE",
                "is_obstructed": False,
                "is_offline": True,
                "reason": "Frozen video feed detected (> 2.0s bit-identical)"
            }

        self.is_offline = False

        # Initialize baseline on first valid frame if not calibrated
        if not self.has_calibrated_baseline:
            self.calibrate_baseline(frame_bgr)

        # 2. Metric extraction
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        img_var = max(1.0, float(np.var(gray)))
        rel_sharp = sharp / img_var

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        bright = float(hsv[..., 2].mean() / 255.0)
        self.bright_history.append((now, bright))

        med = float(np.median(gray))
        low_th = max(10, int(0.66 * med))
        high_th = min(250, int(1.33 * med))
        edges = cv2.Canny(gray, low_th, high_th) > 0
        h, w = edges.shape
        tiles = [
            float(edges[i * h // 4 : (i + 1) * h // 4, j * w // 4 : (j + 1) * w // 4].mean())
            for i in range(4)
            for j in range(4)
        ]

        # 3. Fail condition tests
        # A. Sharpness: < 0.35 * baseline and relative edge loss < 0.50 * baseline
        base_rel = getattr(self, "baseline_rel_sharp", 0.20)
        sharp_fail = (sharp < (0.35 * self.baseline_sharp)) and (rel_sharp < (0.50 * base_rel))

        # B. Brightness: < 0.08 (pitch black) or severe drop with lost relative edge structure
        bright_drop = False
        for old_t, old_b in self.bright_history:
            if now - old_t <= 2.0 and old_b > 0.05:
                if (old_b - bright) / old_b > 0.60:
                    bright_drop = True
                    break
        bright_fail = (bright < 0.08) or (bright_drop and rel_sharp < (0.50 * base_rel))

        # C. Texture: > 60% of tiles fall below 0.3 * baseline
        low_texture_tiles = sum(
            t < (0.30 * b) for t, b in zip(tiles, self.baseline_tiles)
        )
        texture_fail = (low_texture_tiles / 16.0) > 0.60

        # D. Landmark visibility: < 40% matched
        landmark_fail = landmark_ratio < 0.40

        fail_count = sum([sharp_fail, bright_fail, texture_fail, landmark_fail])
        failing_now = fail_count >= 2

        # 4. Temporal hysteresis: 1.5s hold to obstruct, 3.0s pass to release
        if failing_now:
            self.pass_start_time = None
            if self.fail_start_time is None:
                self.fail_start_time = now
            elif (now - self.fail_start_time) >= self.hold_sec:
                self.is_obstructed = True
        else:
            self.fail_start_time = None
            if self.pass_start_time is None:
                self.pass_start_time = now
            elif (now - self.pass_start_time) >= self.release_sec:
                self.is_obstructed = False

        # Soft update baseline when healthy
        if not self.is_obstructed and not failing_now:
            self.baseline_sharp = 0.995 * self.baseline_sharp + 0.005 * sharp
            self.baseline_bright = 0.995 * self.baseline_bright + 0.005 * bright

        status_str = "CAMERA_OBSTRUCTED" if self.is_obstructed else "HEALTHY"

        return {
            "ok": not self.is_obstructed,
            "status": status_str,
            "is_obstructed": self.is_obstructed,
            "is_offline": False,
            "metrics": {
                "sharpness": round(sharp, 2),
                "sharpness_baseline": round(self.baseline_sharp, 2),
                "sharp_fail": sharp_fail,
                "brightness": round(bright, 3),
                "bright_fail": bright_fail,
                "low_texture_ratio": round(low_texture_tiles / 16.0, 2),
                "texture_fail": texture_fail,
                "landmark_ratio": round(landmark_ratio, 2),
                "landmark_fail": landmark_fail,
                "fail_count": fail_count,
            }
        }
