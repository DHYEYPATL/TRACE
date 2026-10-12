"""
Alignment and Camera-Bump Detection module.
Maintains homography registration using ArUco markers and ORB landmarks.
Separates legitimate camera drift from physical camera tampering (Section 5.3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("trace.align")


class FrameAligner:
    """
    Tracks static environmental landmarks to correct camera drift or detect tampering.
    """

    def __init__(self, reference_path: str = "data/reference_frame.jpg"):
        # ArUco detector (Dictionary 4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # ORB feature detector for fallback/complementary landmarks
        self.orb = cv2.ORB_create(nfeatures=500)

        # Reference anchor data
        self.reference_path: str = reference_path
        self.ref_frame_gray: Optional[np.ndarray] = None
        self.ref_aruco_corners: Dict[int, np.ndarray] = {}
        self.ref_orb_kps: Optional[List[cv2.KeyPoint]] = None
        self.ref_orb_des: Optional[np.ndarray] = None

        # Current state
        self.current_homography: np.ndarray = np.eye(3, dtype=np.float32)
        self.mean_shift_px: float = 0.0
        self.inlier_ratio: float = 1.0
        self.is_tampered: bool = False

    def save_reference(self, path: Optional[str] = None, frame_bgr: Optional[np.ndarray] = None) -> bool:
        """Persists approved reference frame to disk."""
        target_path = Path(path or self.reference_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        img_to_save = frame_bgr if frame_bgr is not None else self.ref_frame_gray
        if img_to_save is not None and img_to_save.size > 0:
            ok = cv2.imwrite(str(target_path), img_to_save)
            if ok:
                logger.info(f"Saved approved reference frame to {target_path}")
            return bool(ok)
        return False

    def load_reference(self, path: Optional[str] = None) -> bool:
        """Loads approved reference frame and extracts baseline landmarks without resetting camera state."""
        target_path = Path(path or self.reference_path)
        if not target_path.exists():
            return False
        img = cv2.imread(str(target_path))
        if img is None or img.size == 0:
            return False
        self.register_reference(img, save_to_disk=False, force=True)
        logger.info(f"Restored approved reference frame from {target_path}")
        return True

    def register_reference(self, frame_bgr: np.ndarray, save_to_disk: bool = True, force: bool = False) -> Dict[str, Any]:
        """Establishes reference frame and extracts baseline landmarks. Does not overwrite if already registered unless forced."""
        if self.ref_frame_gray is not None and not force:
            return {
                "aruco_markers_found": len(self.ref_aruco_corners),
                "orb_features_found": len(self.ref_orb_kps) if self.ref_orb_kps else 0,
                "already_registered": True
            }

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.ref_frame_gray = gray

        # Detect reference ArUco markers (restricted to wall landmarks IDs 1-3)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        self.ref_aruco_corners.clear()
        if ids is not None:
            for marker_id, c in zip(ids.flatten(), corners):
                m_id = int(marker_id)
                if m_id in (1, 2, 3):
                    self.ref_aruco_corners[m_id] = c.reshape(-1, 2)

        # Extract reference ORB features
        kps, des = self.orb.detectAndCompute(gray, None)
        self.ref_orb_kps = kps
        self.ref_orb_des = des

        if save_to_disk:
            self.save_reference(frame_bgr=frame_bgr)

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

        # 1. Match ArUco wall landmarks (strictly IDs 1-3)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is not None:
            for marker_id, c in zip(ids.flatten(), corners):
                m_id = int(marker_id)
                if m_id in (1, 2, 3) and m_id in self.ref_aruco_corners:
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

    def warp_to_reference(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Warps the current frame into the canonical reference coordinate system
        using the estimated homography (mapping current -> reference).
        """
        if self.ref_frame_gray is None or self.current_homography is None or self.is_tampered:
            return frame_bgr
        if self.mean_shift_px < 3.0:
            return frame_bgr
        h, w = self.ref_frame_gray.shape[:2]
        try:
            return cv2.warpPerspective(frame_bgr, self.current_homography, (w, h), flags=cv2.INTER_LINEAR)
        except Exception:
            return frame_bgr

    def transform_polygon(self, polygon: List[List[int]]) -> List[List[int]]:
        """
        Maps reference-frame polygon into current-frame coordinates using inverse homography.
        """
        if self.mean_shift_px < 3.0 or self.is_tampered or self.current_homography is None:
            return polygon

        pts = np.float32(polygon).reshape(-1, 1, 2)
        try:
            # self.current_homography maps current -> reference; its inverse maps reference -> current
            H_inv = np.linalg.inv(self.current_homography)
            warped = cv2.perspectiveTransform(pts, H_inv)
            return [[int(round(p[0][0])), int(round(p[0][1]))] for p in warped]
        except Exception:
            return polygon
