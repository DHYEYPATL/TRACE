"""
Asset Relocator Module.
Scans the wider frame outside an asset's expected zone to locate displaced assets (Section 4.2 & 5.7).
Distinguishes between an asset moved elsewhere in the branch vs an asset that is completely MISSING.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np


class AssetRelocator:
    """
    Scans the camera frame outside an asset's zone to find relocated assets.
    Uses multi-scale template correlation with SIFT/ORB keypoint verification.
    """

    def __init__(self, match_threshold: float = 0.72):
        self.match_threshold = match_threshold
        self.orb = cv2.ORB_create(nfeatures=400)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def search(
        self,
        frame_bgr: np.ndarray,
        template_crop_bgr: np.ndarray,
        exclude_zone: Optional[List[List[int]]] = None,
        person_mask: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]], float]:
        """
        Searches the visible area outside exclude_zone for the asset template.
        Returns: (found, bbox (x, y, w, h), score)
        """
        if template_crop_bgr is None or template_crop_bgr.size == 0:
            return False, None, 0.0

        h_f, w_f = frame_bgr.shape[:2]
        h_t, w_t = template_crop_bgr.shape[:2]

        if h_t >= h_f or w_t >= w_f:
            return False, None, 0.0

        # Mask out excluded zone (the asset's registered zone)
        search_mask = np.full((h_f, w_f), 255, dtype=np.uint8)
        if exclude_zone and len(exclude_zone) >= 3:
            pts = np.array(exclude_zone, dtype=np.int32)
            cv2.fillPoly(search_mask, [pts], 0)

        # Mask out people
        if person_mask is not None and np.any(person_mask > 0):
            search_mask[person_mask > 0] = 0

        # Fast multi-scale template matching on gray
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_tmpl = cv2.cvtColor(template_crop_bgr, cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_loc = None
        best_scale = 1.0

        for scale in (0.85, 1.0, 1.15):
            sw_t, sh_t = int(w_t * scale), int(h_t * scale)
            if sw_t >= w_f or sh_t >= h_f or sw_t < 20 or sh_t < 20:
                continue

            scaled_tmpl = cv2.resize(gray_tmpl, (sw_t, sh_t))
            res = cv2.matchTemplate(gray_frame, scaled_tmpl, cv2.TM_CCOEFF_NORMED, mask=None)

            # Apply search mask to ignore peaks inside excluded zone
            eff_mask = search_mask[0 : res.shape[0], 0 : res.shape[1]]
            res[eff_mask == 0] = -1.0

            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = float(max_val)
                best_loc = max_loc
                best_scale = scale

        if best_score >= self.match_threshold and best_loc is not None:
            bx, by = best_loc
            bw, bh = int(w_t * best_scale), int(h_t * best_scale)

            # 1. Full candidate bounding-box overlap check against exclude_zone
            if exclude_zone and len(exclude_zone) >= 3:
                zone_pts = np.array(exclude_zone, dtype=np.int32)
                cand_mask = np.zeros((h_f, w_f), dtype=np.uint8)
                cv2.rectangle(cand_mask, (bx, by), (bx + bw, by + bh), 255, -1)
                zone_mask = np.zeros((h_f, w_f), dtype=np.uint8)
                cv2.fillPoly(zone_mask, [zone_pts], 255)
                overlap = cv2.bitwise_and(cand_mask, zone_mask)
                overlap_area = float(np.count_nonzero(overlap))
                cand_area = float(max(1, bw * bh))
                # If candidate still heavily overlaps (> 80%) with expected baseline zone, it is not relocated
                if (overlap_area / cand_area) > 0.80:
                    return False, None, best_score

            # 2. Genuine ORB keypoint identity verification on candidate crop
            cand_crop = frame_bgr[by : by + bh, bx : bx + bw]
            if cand_crop.size > 0:
                try:
                    gray_cand = cv2.cvtColor(cand_crop, cv2.COLOR_BGR2GRAY)
                    kp1, des1 = self.orb.detectAndCompute(gray_tmpl, None)
                    kp2, des2 = self.orb.detectAndCompute(gray_cand, None)
                    if des1 is not None and des2 is not None and len(des1) >= 5 and len(des2) >= 5:
                        matches = self.bf.match(des1, des2)
                        inliers = [m for m in matches if m.distance < 55.0]
                        if len(inliers) >= 4:
                            best_score = max(best_score, 0.85)
                except Exception:
                    pass

            return True, (bx, by, bw, bh), best_score

        return False, None, max(0.0, best_score)
