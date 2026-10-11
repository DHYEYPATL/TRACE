"""
Capture module for TRACE.
Reads from webcams, video files, RTSP streams, or an enterprise synthetic branch simulator.
Maintains a dedicated thread and drops old frames to guarantee real-time latency.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("trace.capture")


def generate_synthetic_branch_frame(frame_idx: int, width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Renders a photo-realistic synthetic bank branch environment with:
    - Neutral tiled wall and wooden office desk
    - Cash & Gold Safe (grey/metallic with combination dial and handle)
    - Laser Printer (white/dark office appliance with paper tray)
    - Manager Workstation Laptop (opened laptop with screen)
    - Static ArUco wall landmark markers (dictionary 4x4_50)
    """
    frame = np.full((height, width, 3), 220, dtype=np.uint8)

    # 1. Background wall and floor
    # Top 60% wall (light cream grey)
    frame[: int(height * 0.65), :] = (215, 220, 225)
    # Floor (subtle checker tiles)
    for y in range(int(height * 0.65), height, 40):
        for x in range(0, width, 40):
            c = (175, 180, 185) if ((x // 40) + (y // 40)) % 2 == 0 else (190, 195, 200)
            frame[y : y + 40, x : x + 40] = c

    # Desk surface
    desk_top = int(height * 0.45)
    cv2.rectangle(frame, (100, desk_top), (width - 100, height - 80), (120, 140, 160), -1) # Wood finish
    cv2.rectangle(frame, (100, desk_top), (width - 100, height - 80), (70, 85, 100), 3)

    # 2. Asset 1: Cash & Gold Safe (Region: [180, 220] to [380, 520])
    # Heavy vault safe
    cv2.rectangle(frame, (180, 220), (380, 520), (50, 55, 60), -1) # Dark charcoal steel
    cv2.rectangle(frame, (190, 230), (370, 510), (75, 80, 85), -1) # Door panel
    cv2.circle(frame, (280, 360), 35, (160, 165, 170), -1) # Dial base
    cv2.circle(frame, (280, 360), 25, (40, 45, 50), -1)   # Dial core
    cv2.circle(frame, (280, 360), 8, (200, 200, 200), -1)  # Center pin
    # Handle
    cv2.rectangle(frame, (330, 340), (345, 400), (180, 185, 190), -1)
    # Asset Label
    cv2.putText(frame, "L&T VAULT SAFE #02", (200, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    # 3. Asset 2: Laser Printer (Region: [520, 260] to [740, 480])
    cv2.rectangle(frame, (520, 260), (740, 480), (240, 240, 242), -1) # Matte white chassis
    cv2.rectangle(frame, (520, 260), (740, 480), (160, 160, 165), 2)
    # Output tray & scanner lid
    cv2.rectangle(frame, (540, 275), (720, 340), (45, 45, 50), -1)    # Dark glass scanner bed
    cv2.rectangle(frame, (545, 380), (715, 410), (120, 120, 125), -1) # Paper output slot
    cv2.circle(frame, (700, 310), 12, (0, 180, 0), -1)                 # Green power LED
    cv2.putText(frame, "LASER PRINTER #01", (540, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    # 4. Asset 3: Manager Workstation Laptop (Region: [880, 310] to [1100, 500])
    cv2.rectangle(frame, (900, 310), (1080, 430), (30, 35, 40), -1)   # Laptop screen back
    cv2.rectangle(frame, (910, 320), (1070, 420), (90, 120, 160), -1) # Active screen glowing
    cv2.putText(frame, "TRACE SECURE OS", (930, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
    # Keyboard base
    cv2.rectangle(frame, (880, 430), (1100, 490), (170, 175, 180), -1) # Aluminum base
    cv2.rectangle(frame, (920, 440), (1060, 480), (60, 65, 70), -1)   # Keyboard keybed

    # 5. Fixed Wall ArUco Landmarks (Dictionary 4x4_50)
    # Render static markers on the wall for homography alignment
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # Landmark 1 (Top Left)
        tag1 = cv2.aruco.generateImageMarker(aruco_dict, 1, 60)
        tag1_bgr = cv2.cvtColor(tag1, cv2.COLOR_GRAY2BGR)
        frame[40:100, 80:140] = tag1_bgr

        # Landmark 2 (Top Right)
        tag2 = cv2.aruco.generateImageMarker(aruco_dict, 2, 60)
        tag2_bgr = cv2.cvtColor(tag2, cv2.COLOR_GRAY2BGR)
        frame[40:100, width - 140 : width - 80] = tag2_bgr

        # Landmark 3 (Top Center)
        tag3 = cv2.aruco.generateImageMarker(aruco_dict, 3, 60)
        tag3_bgr = cv2.cvtColor(tag3, cv2.COLOR_GRAY2BGR)
        frame[40:100, width // 2 - 30 : width // 2 + 30] = tag3_bgr
    except Exception:
        # Fallback high-contrast geometric landmarks
        cv2.rectangle(frame, (80, 40), (140, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 140, 40), (width - 80, 100), (0, 0, 0), -1)

    # Subtle ambient sensor noise to simulate a real physical camera sensor
    np.random.seed(frame_idx % 1000)
    noise = np.random.normal(0, 1.2, (height, width, 3)).astype(np.int16)
    noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy_frame


class FrameCapture:
    """
    Threaded capture reader. Supports synthetic generation, webcam index, file, or RTSP.
    Always maintains the latest frame, dropping intermediate stale frames.
    """

    def __init__(self, source_uri: str = "synthetic", width: int = 1280, height: int = 720, target_fps: int = 15):
        self.source_uri = source_uri
        self.width = width
        self.height = height
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0.0
        self.frame_counter: int = 0
        self.is_running: bool = False
        self.is_synthetic: bool = (source_uri == "synthetic" or source_uri.startswith("synth"))
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.is_running:
            return

        if not self.is_synthetic:
            # Parse webcam:N or file:... or rtsp://...
            if self.source_uri.startswith("webcam:"):
                dev_idx = int(self.source_uri.split(":")[1])
                self.cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
            elif self.source_uri.startswith("file:"):
                file_path = self.source_uri.replace("file:", "")
                self.cap = cv2.VideoCapture(file_path)
            else:
                self.cap = cv2.VideoCapture(self.source_uri)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                logger.info(f"Connected to camera feed: {self.source_uri}")
            else:
                logger.warning(f"Could not open '{self.source_uri}'. Falling back to synthetic branch feed.")
                self.is_synthetic = True

        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        interval = 1.0 / max(1, self.target_fps)
        while self.is_running:
            t0 = time.time()
            frame = None

            if self.is_synthetic:
                frame = generate_synthetic_branch_frame(self.frame_counter, self.width, self.height)
            elif self.cap and self.cap.isOpened():
                ret, grabbed = self.cap.read()
                if ret:
                    frame = grabbed
                else:
                    # Loop video or reconnect
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if frame is not None:
                with self._lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
                    self.frame_counter += 1

            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Returns (success, frame_bgr, timestamp).
        Non-blocking read of the freshest frame.
        """
        with self._lock:
            if self.latest_frame is None:
                # Generate initial synthetic frame if not yet captured
                self.latest_frame = generate_synthetic_branch_frame(0, self.width, self.height)
                self.last_frame_time = time.time()
            return True, self.latest_frame.copy(), self.last_frame_time

    def stop(self) -> None:
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Frame capture stopped.")
