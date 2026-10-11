"""
Alignment and Camera-Bump Detection module.
Maintains homography registration using ArUco markers and ORB landmarks.
Separates legitimate camera drift from physical camera tampering (Section 5.3).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("trace.align")


class FrameAligner:
    """
    Tracks static environmental landmarks to correct camera drift or detect tampering.
    """

    def __init__(self):
        # ArUco detector (Dictionary 4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # ORB feature detector for fallback/complementary landmarks
        self.orb = cv2.ORB_create(nfeatures=500)

        # Reference anchor data
        self.ref_frame_gray: Optional[np.ndarray] = None
        self.ref_aruco_corners: Dict[int, np.ndarray] = {}
        self.ref_orb_kps: Optional[List[cv2.KeyPoint]] = None
        self.ref_orb_des: Optional[np.ndarray] = None

        # Current state
        self.current_homography: np.ndarray = np.eye(3, dtype=np.float32)
        self.mean_shift_px: float = 0.0
        self.inlier_ratio: float = 1.0
        self.is_tampered: bool = False

    def register_reference(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Establishes reference frame and extracts baseline landmarks."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.ref_frame_gray = gray

        # Detect reference ArUco markers
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        self.ref_aruco_corners.clear()
        if ids is not None:
            for marker_id, c in zip(ids.flatten(), corners):
                self.ref_aruco_corners[int(marker_id)] = c.reshape(-1, 2)

        # Extract reference ORB features
        kps, des = self.orb.detectAndCompute(gray, None)
        self.ref_orb_kps = kps
        self.ref_orb_des = des

        return {
            "aruco_markers_found": len(self.ref_aruco_corners),
            "orb_features_found": len(kps) if kps else 0,
        }

    def align_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Calculates homography between current frame and reference.
        Determines if camera has drifted or been physically tampered.
        """
        if self.ref_frame_gray is None:
            self.register_reference(frame_bgr)
            return {"status": "REGISTERED", "shift_px": 0.0, "is_tampered": False}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        src_pts: List[np.ndarray] = []
        dst_pts: List[np.ndarray] = []

        # 1. Match ArUco markers
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is not None:
            for marker_id, c in zip(ids.flatten(), corners):
                m_id = int(marker_id)
                if m_id in self.ref_aruco_corners:
                    ref_c = self.ref_aruco_corners[m_id]
                    curr_c = c.reshape(-1, 2)
                    for r_pt, cur_pt in zip(ref_c, curr_c):
                        src_pts.append(cur_pt)
                        dst_pts.append(r_pt)

        # 2. Match ORB background features if ArUco markers are sparse
        if len(src_pts) < 4 and self.ref_orb_des is not None:
            kps, des = self.orb.detectAndCompute(gray, None)
            if des is not None and len(des) > 10 and self.ref_orb_des is not None and len(self.ref_orb_des) > 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des, self.ref_orb_des)
                matches = sorted(matches, key=lambda x: x.distance)[:80]
                for m in matches:
                    src_pts.append(kps[m.queryIdx].pt)
                    dst_pts.append(self.ref_orb_kps[m.trainIdx].pt)

        # Compute Homography
        if len(src_pts) >= 4:
            src_arr = np.float32(src_pts)
            dst_arr = np.float32(dst_pts)

            H, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
            if H is not None and mask is not None:
                inliers = int(mask.sum())
                total = len(src_pts)
                self.inlier_ratio = float(inliers / max(1, total))

                # Calculate average pixel displacement
                diff = np.linalg.norm(src_arr - dst_arr, axis=1)
                self.mean_shift_px = float(np.mean(diff))
                self.current_homography = H

                # Section 5.3 rules:
                # < 3px: ignore
                # 3 to 25px: warp regions (camera drift)
                # > 25px or inliers < 30%: TAMPERED (camera knocked)
                if self.mean_shift_px > 25.0 or self.inlier_ratio < 0.30:
                    self.is_tampered = True
                else:
                    self.is_tampered = False
            else:
                self.is_tampered = True
        else:
            # Low landmark count
            self.inlier_ratio = 0.2
            self.mean_shift_px = 30.0
            self.is_tampered = True

        status = "TAMPERED" if self.is_tampered else ("DRIFT" if self.mean_shift_px >= 3.0 else "ALIGNED")

        return {
            "status": status,
            "shift_px": round(self.mean_shift_px, 2),
            "inlier_ratio": round(self.inlier_ratio, 2),
            "is_tampered": self.is_tampered,
            "landmark_ratio": self.inlier_ratio
        }

    def transform_polygon(self, polygon: List[List[int]]) -> List[List[int]]:
        """Warps region polygon according to camera drift homography."""
        if self.mean_shift_px < 3.0 or self.is_tampered or self.current_homography is None:
            return polygon

        pts = np.float32(polygon).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(pts, self.current_homography)
            return [[int(round(p[0][0])), int(round(p[0][1]))] for p in warped]
        except Exception:
            return polygon
