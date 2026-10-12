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
                "tag_matched": tag_id is None,
                "tag_status": "NONE" if tag_id is None else "UNREADABLE",
                "detected_tag_id": None,
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
        # Explicit states: MATCH, MISMATCH, UNREADABLE, NONE
        tag_status = "NONE"
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
                    tag_status = "MISMATCH"
                    tag_matched = False
                    fused_score = min(fused_score, 0.2)  # Severe penalty for swapped tag!
                else:
                    tag_status = "MATCH"
                    tag_matched = True
            else:
                tag_status = "UNREADABLE"
                # If an asset tag is expected but absent/unreadable, identity is NOT confirmed
                tag_matched = False

        # 5. Empty-region baseline match (for positive-evidence MISSING)
        empty_match = 0.0
        if empty_roi_bgr is not None:
            gray_live = cv2.cvtColor(live_crop_bgr, cv2.COLOR_BGR2GRAY)
            gray_empty = cv2.cvtColor(empty_roi_bgr, cv2.COLOR_BGR2GRAY)
            empty_match = compute_ssim_grayscale(gray_live, gray_empty)

        # 6. Spatial tile consistency check for partial obstruction detection
        is_partially_occluded = False
        occlusion_ratio = 0.0
        if baseline_median_crop is not None and baseline_median_crop.size > 0:
            h_b, w_b = baseline_median_crop.shape[:2]
            h_l, w_l = live_crop_bgr.shape[:2]
            if h_b >= 16 and w_b >= 16:
                sample_live = live_crop_bgr if (h_b == h_l and w_b == w_l) else cv2.resize(live_crop_bgr, (w_b, h_b))
                gray_l = cv2.cvtColor(sample_live, cv2.COLOR_BGR2GRAY)
                gray_b = cv2.cvtColor(baseline_median_crop, cv2.COLOR_BGR2GRAY)

                grid_r, grid_c = 3, 3
                th, tw = h_b // grid_r, w_b // grid_c
                matching_tiles = 0
                differing_tiles = 0
                total_tiles = grid_r * grid_c

                for r in range(grid_r):
                    for c in range(grid_c):
                        t_l = gray_l[r*th:(r+1)*th, c*tw:(c+1)*tw]
                        t_b = gray_b[r*th:(r+1)*th, c*tw:(c+1)*tw]
                        if t_l.size == 0 or t_b.size == 0:
                            continue
                        tile_ssim = compute_ssim_grayscale(t_l, t_b)
                        if tile_ssim >= 0.70:
                            matching_tiles += 1
                        elif tile_ssim < 0.45:
                            differing_tiles += 1

                # Partial occlusion: some parts of the asset match baseline, while other parts are obscured
                if (matching_tiles >= 2 and differing_tiles >= 2) or (inliers >= 3 and fused_score < 0.65):
                    is_partially_occluded = True
                elif tag_id is not None and tag_status == "UNREADABLE" and differing_tiles >= 1 and (fused_score >= 0.35 or inliers >= 2):
                    is_partially_occluded = True

                occlusion_ratio = float(differing_tiles / total_tiles) if is_partially_occluded else 0.0

        return {
            "present_score": round(float(fused_score), 4),
            "s_emb": round(float(s_emb), 4),
            "s_kp": round(float(s_kp), 4),
            "inliers": inliers,
            "empty_match": round(float(empty_match), 4),
            "tag_matched": tag_matched,
            "tag_status": tag_status,
            "detected_tag_id": actual_detected_tag,
            "is_partially_occluded": is_partially_occluded,
            "occlusion_ratio": round(float(occlusion_ratio), 4),
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


def detect_primary_tag(crop_bgr: np.ndarray, detector: Optional[Any] = None) -> Optional[int]:
    """Detects primary ArUco tag ID in a crop, or None."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if detector is None:
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        except Exception:
            return None
    try:
        corners, ids, _ = detector.detectMarkers(crop_bgr)
        if ids is not None and len(ids) > 0:
            return int(ids[0][0])
    except Exception:
        pass
    return None
