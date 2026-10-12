"""
Region Verification module.
Fuses deep Visual DNA embedding similarity with geometric SIFT/ORB keypoint matching (Section 5.6).
Calculates drift fraction, emptiness proof, and identity verification.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from trace.vision.embed import AssetDescriptor, LabSobelDescriptorV1, VisualDNAExtractor
from trace.vision.gate import compute_ssim_grayscale


class RegionVerifier:
    """
    Verifies registered assets against baseline profiles, calculating confidence,
    centroid drift, and positive emptiness evidence.
    """

    def __init__(self, descriptor: Optional[AssetDescriptor] = None):
        self.extractor = descriptor if descriptor is not None else LabSobelDescriptorV1()
        # Keypoint matcher using SIFT or ORB
        try:
            self.detector = cv2.SIFT_create()
            self.detector_name = "SIFT"
            self.norm_type = cv2.NORM_L2
        except Exception:
            self.detector = cv2.ORB_create(nfeatures=500)
            self.detector_name = "ORB"
            self.norm_type = cv2.NORM_HAMMING

        self.bf = cv2.BFMatcher(self.norm_type)
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        except Exception:
            self.aruco_detector = None

    def get_model_metadata(self) -> Dict[str, Any]:
        """Reports honest model identity, feature dimensions, and matcher types."""
        desc_meta = self.extractor.get_metadata()
        return {
            "descriptor": desc_meta,
            "keypoint_detector": self.detector_name,
            "model_version": f"{desc_meta['name']}-{desc_meta['version']}+{self.detector_name}",
            "device": "cpu",
            "inference_ready": True,
        }

    def verify_asset(
        self,
        live_crop_bgr: np.ndarray,
        baseline_variants: List[np.ndarray],
        baseline_median_crop: np.ndarray,
        empty_roi_bgr: Optional[np.ndarray] = None,
        tag_id: Optional[int] = None,
        detected_tag_id: Optional[int] = None,
        low_texture_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Executes fused region verification matching Section 5.6.
        """
        if live_crop_bgr is None or live_crop_bgr.size == 0:
            return {
                "present_score": 0.0,
                "s_emb": 0.0,
                "s_kp": 0.0,
                "inliers": 0,
                "empty_match": 1.0,
                "tag_matched": True,
            }

        # 1. Embedding score s_emb: mean of top 3 cosine similarities to stored variants
        live_vec = self.extractor.extract(live_crop_bgr)
        sims = [self.extractor.cosine_similarity(live_vec, var_vec) for var_vec in baseline_variants]
        sims_sorted = sorted(sims, reverse=True)
        top3 = sims_sorted[:3] if len(sims_sorted) >= 3 else sims_sorted
        s_emb = float(np.mean(top3)) if top3 else 0.0

        # 2. Keypoint score s_kp (RANSAC inlier count against baseline median)
        inliers = 0
        s_kp = 0.0
        if not low_texture_mode and baseline_median_crop is not None:
            gray_live = cv2.cvtColor(live_crop_bgr, cv2.COLOR_BGR2GRAY)
            gray_base = cv2.cvtColor(baseline_median_crop, cv2.COLOR_BGR2GRAY)

            kp1, des1 = self.detector.detectAndCompute(gray_live, None)
            kp2, des2 = self.detector.detectAndCompute(gray_base, None)

            if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
                try:
                    matches = self.bf.knnMatch(des1, des2, k=2)
                    good = []
                    for m_pair in matches:
                        if len(m_pair) == 2:
                            m, n = m_pair
                            if m.distance < 0.75 * n.distance:
                                good.append(m)

                    if len(good) >= 4:
                        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
                        if mask is not None:
                            inliers = int(mask.sum())
                except Exception:
                    inliers = 0

            s_kp = float(min(1.0, inliers / 15.0))

        # 3. Fusion score
        if low_texture_mode or inliers < 4:
            fused_score = s_emb
        else:
            fused_score = 0.6 * s_emb + 0.4 * s_kp

        # 4. ArUco / QR tag check
        tag_matched = True
        actual_detected_tag = detected_tag_id
        if tag_id is not None:
            if actual_detected_tag is None and self.aruco_detector is not None:
                try:
                    gray_crop = cv2.cvtColor(live_crop_bgr, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = self.aruco_detector.detectMarkers(gray_crop)
                    if ids is not None and len(ids) > 0:
                        actual_detected_tag = int(ids[0][0])
                except Exception:
                    actual_detected_tag = None

            if actual_detected_tag is not None:
                if actual_detected_tag != tag_id:
                    tag_matched = False
                    fused_score = min(fused_score, 0.2)  # Severe penalty for swapped tag!
                else:
                    tag_matched = True

        # 5. Empty-region baseline match (for positive-evidence MISSING)
        empty_match = 0.0
        if empty_roi_bgr is not None:
            gray_live = cv2.cvtColor(live_crop_bgr, cv2.COLOR_BGR2GRAY)
            gray_empty = cv2.cvtColor(empty_roi_bgr, cv2.COLOR_BGR2GRAY)
            empty_match = compute_ssim_grayscale(gray_live, gray_empty)

        return {
            "present_score": round(float(fused_score), 4),
            "s_emb": round(float(s_emb), 4),
            "s_kp": round(float(s_kp), 4),
            "inliers": inliers,
            "empty_match": round(float(empty_match), 4),
            "tag_matched": tag_matched,
        }

    @staticmethod
    def calculate_centroid_drift(
        current_region: List[List[int]],
        baseline_region: List[List[int]]
    ) -> Tuple[float, float]:
        """
        Calculates drift distance in pixels and drift fraction relative to diagonal.
        Returns: (drift_px, drift_frac)
        """
        cur_pts = np.array(current_region, dtype=np.float32)
        base_pts = np.array(baseline_region, dtype=np.float32)

        cur_cx, cur_cy = float(np.mean(cur_pts[:, 0])), float(np.mean(cur_pts[:, 1]))
        base_cx, base_cy = float(np.mean(base_pts[:, 0])), float(np.mean(base_pts[:, 1]))

        drift_px = math.sqrt((cur_cx - base_cx) ** 2 + (cur_cy - base_cy) ** 2)

        # Baseline diagonal
        rx, ry, rw, rh = cv2.boundingRect(np.int32(base_pts))
        diag = max(1.0, math.sqrt(rw ** 2 + rh ** 2))
        drift_frac = drift_px / diag

        return drift_px, drift_frac
