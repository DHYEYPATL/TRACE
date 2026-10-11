"""
Demo Scenario Fault Injector.
Applies synthetic faults directly to the raw frame stream before the pipeline,
allowing rigorous testing of edge cases (lens cover, lighting, occlusion, swaps, tampering)
matching Section 8.6 and the Round 2 Twist Playbook.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class DemoFaultInjector:
    """Injects live scenario perturbations into video frames before processing."""

    def __init__(self):
        self.faults: Dict[str, Any] = {
            "cover_lens": {"active": False, "until": 0.0},
            "dim_lights": {"active": False, "until": 0.0},
            "occlude_asset": {"active": False, "asset_id": None, "until": 0.0},
            "cut_network": {"active": False},
            "move_asset": {"active": False, "asset_id": None, "dx": 60, "dy": 40},
            "remove_asset": {"active": False, "asset_id": None},
            "place_unknown_object": {"active": False},
            "swap_asset": {"active": False, "asset_id": None},
            "open_safe_door": {"active": False, "asset_id": "safe-02"},
            "knock_camera": {"active": False, "until": 0.0, "dx": 30, "dy": 20},
        }

    def trigger_fault(self, fault_type: str, asset_id: Optional[str] = None, duration_s: float = 6.0) -> Dict[str, Any]:
        """Activates a specific fault for testing."""
        now = time.time()
        if fault_type not in self.faults:
            return {"ok": False, "error": f"Unknown fault type: {fault_type}"}

        target = self.faults[fault_type]
        target["active"] = True
        if "until" in target:
            target["until"] = now + duration_s
        if "asset_id" in target:
            target["asset_id"] = asset_id

        return {"ok": True, "fault": fault_type, "asset_id": asset_id, "duration": duration_s}

    def reset_all(self) -> Dict[str, Any]:
        """Clears all active perturbations."""
        for k in self.faults:
            self.faults[k]["active"] = False
            if "until" in self.faults[k]:
                self.faults[k]["until"] = 0.0
            if "asset_id" in self.faults[k]:
                self.faults[k]["asset_id"] = None
        return {"ok": True, "message": "All demo faults reset to normal"}

    def is_network_cut(self) -> bool:
        return bool(self.faults["cut_network"]["active"])

    def set_network_cut(self, cut: bool) -> None:
        self.faults["cut_network"]["active"] = cut

    def apply_to_frame(
        self,
        frame_bgr: np.ndarray,
        assets: Dict[str, Any],
        empty_rois: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms incoming video frame based on active perturbations.
        Returns: (modified_frame_bgr, synthetic_person_mask)
        """
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        person_mask = np.zeros((h, w), dtype=np.uint8)
        now = time.time()

        # 1. Knock Camera (Homography shift)
        knock = self.faults["knock_camera"]
        if knock["active"]:
            if knock["until"] > 0 and now > knock["until"]:
                knock["active"] = False
            else:
                M = np.float32([[1, 0, knock["dx"]], [0, 1, knock["dy"]]])
                out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 2. Dim Lights (Drop illumination to 20%)
        dim = self.faults["dim_lights"]
        if dim["active"]:
            if dim["until"] > 0 and now > dim["until"]:
                dim["active"] = False
            else:
                out = (out.astype(np.float32) * 0.22).astype(np.uint8)

        # 3. Cover Lens (Complete blur + dark vignette)
        cover = self.faults["cover_lens"]
        if cover["active"]:
            if cover["until"] > 0 and now > cover["until"]:
                cover["active"] = False
            else:
                out = cv2.GaussianBlur(out, (71, 71), 0)
                out = (out.astype(np.float32) * 0.12).astype(np.uint8)

        # 4. Occlude Asset (Synthetic person overlay on the asset)
        occlude = self.faults["occlude_asset"]
        if occlude["active"]:
            if occlude["until"] > 0 and now > occlude["until"]:
                occlude["active"] = False
            else:
                target_id = occlude["asset_id"]
                target_asset = assets.get(target_id) if target_id else (list(assets.values())[0] if assets else None)
                if target_asset and target_asset.region:
                    pts = np.array(target_asset.region, dtype=np.int32)
                    rx, ry, rw, rh = cv2.boundingRect(pts)
                    # Create person silhouette covering region
                    cv2.ellipse(person_mask, (rx + rw // 2, ry + rh // 2), (rw // 2 + 10, rh // 2 + 15), 0, 0, 360, 255, -1)
                    # Render person on screen in blue/gray clothes
                    cv2.ellipse(out, (rx + rw // 2, ry + rh // 2), (rw // 2 + 10, rh // 2 + 15), 0, 0, 360, (75, 60, 60), -1)

        # 5. Move Asset (Displace the visual asset)
        move = self.faults["move_asset"]
        if move["active"]:
            target_id = move["asset_id"]
            target_asset = assets.get(target_id) if target_id else (list(assets.values())[0] if assets else None)
            if target_asset and target_asset.region:
                pts = np.array(target_asset.region, dtype=np.int32)
                rx, ry, rw, rh = cv2.boundingRect(pts)
                crop = frame_bgr[ry:ry+rh, rx:rx+rw].copy()
                # Clear original location with empty background or neutral color
                out[ry:ry+rh, rx:rx+rw] = (140, 140, 140)
                # Paste at displaced coordinates
                nx, ny = min(w - rw, rx + move["dx"]), min(h - rh, ry + move["dy"])
                out[ny:ny+rh, nx:nx+rw] = crop

        # 6. Remove Asset (Positive-evidence test: replace region with empty baseline background)
        remove = self.faults["remove_asset"]
        if remove["active"]:
            target_id = remove["asset_id"]
            target_asset = assets.get(target_id) if target_id else (list(assets.values())[0] if assets else None)
            if target_asset and target_asset.region:
                pts = np.array(target_asset.region, dtype=np.int32)
                rx, ry, rw, rh = cv2.boundingRect(pts)
                if empty_rois and target_asset.id in empty_rois:
                    e_crop = empty_rois[target_asset.id]
                    if e_crop.shape[:2] == (rh, rw):
                        out[ry:ry+rh, rx:rx+rw] = e_crop
                    else:
                        out[ry:ry+rh, rx:rx+rw] = cv2.resize(e_crop, (rw, rh))
                else:
                    # Neutral background wall/desk color
                    out[ry:ry+rh, rx:rx+rw] = (150, 150, 150)

        # 7. Swap Asset (Replace asset with visually mismatched object / wrong color)
        swap = self.faults["swap_asset"]
        if swap["active"]:
            target_id = swap["asset_id"]
            target_asset = assets.get(target_id) if target_id else (list(assets.values())[0] if assets else None)
            if target_asset and target_asset.region:
                pts = np.array(target_asset.region, dtype=np.int32)
                rx, ry, rw, rh = cv2.boundingRect(pts)
                # Draw lookalike with distinct striped pattern & color
                fake_crop = np.zeros((rh, rw, 3), dtype=np.uint8)
                fake_crop[:] = (30, 130, 220) # Bright orange lookalike
                cv2.line(fake_crop, (0, 0), (rw, rh), (255, 255, 255), 4)
                cv2.putText(fake_crop, "REPLACED", (10, rh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                out[ry:ry+rh, rx:rx+rw] = fake_crop

        # 8. Place Unknown Object (NEW_OBJECT: persistent foreign box in zone)
        new_obj = self.faults["place_unknown_object"]
        if new_obj["active"]:
            # Place foreign parcel between safe and printer
            bx, by, bw, bh = 420, 360, 90, 80
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (40, 90, 180), -1)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (20, 50, 100), 2)
            cv2.putText(out, "UNKNOWN BOX", (bx + 5, by + bh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # 9. Open Safe Door (STATE_CHANGED: safe door swung open, gold visible)
        door = self.faults["open_safe_door"]
        if door["active"]:
            target_asset = assets.get("safe-02")
            if target_asset and target_asset.region:
                pts = np.array(target_asset.region, dtype=np.int32)
                rx, ry, rw, rh = cv2.boundingRect(pts)
                # Render open safe interior
                out[ry:ry+rh, rx:rx+rw] = (25, 25, 25) # Dark cavity
                cv2.line(out, (rx + 15, ry + int(rh * 0.35)), (rx + rw - 30, ry + int(rh * 0.35)), (110, 110, 110), 3)
                cv2.line(out, (rx + 15, ry + int(rh * 0.70)), (rx + rw - 30, ry + int(rh * 0.70)), (110, 110, 110), 3)
                # Gold bar stack on shelf
                cv2.rectangle(out, (rx + 35, ry + int(rh * 0.22)), (rx + 90, ry + int(rh * 0.33)), (40, 180, 230), -1)
                cv2.putText(out, "GOLD", (rx + 42, ry + int(rh * 0.30)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
                # Open door swung to the right
                cv2.rectangle(out, (rx + rw - 25, ry + 10), (rx + rw - 5, ry + rh - 10), (85, 90, 95), -1)

        return out, person_mask

