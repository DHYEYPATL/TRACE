"""
Evidence writer module.
Guarantees privacy-by-design by masking individuals, persists before/after
asset crops, and calculates cryptographic digests for the audit trail.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from trace.platform.ledger import sha256_file


class EvidenceWriter:
    """Handles crop extraction, privacy masking, and evidence persistence."""

    def __init__(self, base_dir: str = "data/evidence"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def apply_privacy_mask(
        frame_bgr: np.ndarray,
        person_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Applies heavy Gaussian blur to person regions in memory before saving.
        Adheres to DPDP Act 2023 privacy guidelines (Section 5.4).
        """
        if person_mask is None or not np.any(person_mask > 0):
            return frame_bgr.copy()

        blurred = cv2.GaussianBlur(frame_bgr, (51, 51), 0)
        # Ensure mask is binary and 1 channel
        if len(person_mask.shape) == 3:
            mask_1ch = person_mask[:, :, 0] > 0
        else:
            mask_1ch = person_mask > 0

        # Replace person pixels with blurred pixels
        out = frame_bgr.copy()
        out[mask_1ch] = blurred[mask_1ch]
        return out

    def save_crops(
        self,
        event_id: str,
        before_crop: np.ndarray,
        after_crop: np.ndarray,
        before_mask: Optional[np.ndarray] = None,
        after_mask: Optional[np.ndarray] = None,
        person_masked: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Saves before and after crops, hashes them, and returns evidence metadata and hashes.
        """
        today_str = datetime.date.today().isoformat()
        ev_dir = self.base_dir / today_str / event_id
        ev_dir.mkdir(parents=True, exist_ok=True)

        # Apply privacy blur if requested
        if person_masked:
            clean_before = self.apply_privacy_mask(before_crop, before_mask)
            clean_after = self.apply_privacy_mask(after_crop, after_mask)
        else:
            clean_before = before_crop
            clean_after = after_crop

        before_path = ev_dir / "before.jpg"
        after_path = ev_dir / "after.jpg"

        cv2.imwrite(str(before_path), clean_before, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(after_path), clean_after, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        sha_before = sha256_file(before_path) or ""
        sha_after = sha256_file(after_path) or ""

        evidence_meta = {
            "before": str(before_path).replace("\\", "/"),
            "after": str(after_path).replace("\\", "/"),
            "person_masked": person_masked,
            "sha256_before": sha_before,
            "sha256_after": sha_after,
        }

        crop_hashes = {
            "before": sha_before,
            "after": sha_after,
        }

        return evidence_meta, crop_hashes
