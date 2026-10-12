"""
Ablation Baseline Models for Real Frame-Level Evaluation.
All models process actual visual frames (np.ndarray) without label leakage,
demonstrating the exact empirical impact of each architectural stage.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

import math
from trace.logic.fsm import AssetFSM, FSMConfig, Obs, S
from trace.logic.policy import PolicyEngine
from trace.vision.align import FrameAligner
from trace.vision.embed import LabSobelDescriptorV1
from trace.vision.gate import compute_ssim_grayscale
from trace.vision.health import CameraHealthMonitor
from trace.vision.people import PeopleSegmenter
from trace.vision.relocate import AssetRelocator
from trace.vision.verify import RegionVerifier


class NaivePixelDiffBaseline:
    """
    Variant 1: Naive pixel-difference baseline.
    Computes mean absolute pixel difference in the asset's bounding box against the baseline crop.
    Fires an immediate theft alarm whenever pixel difference exceeds threshold for > 1.5s.
    Fails on: occlusion, lighting changes, camera bumps. Blind to identity swaps.
    """

    def __init__(self, diff_threshold: float = 30.0, hold_s: float = 1.5):
        self.diff_threshold = diff_threshold
        self.hold_s = hold_s
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.since: Dict[str, Optional[float]] = {}

    def reset(self) -> None:
        self.assets.clear()
        self.since.clear()

    def register_asset(
        self,
        asset_id: str,
        region: List[List[int]],
        baseline_crop: np.ndarray,
        empty_crop: Optional[np.ndarray] = None
    ) -> None:
        pts = np.array(region, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        self.assets[asset_id] = {
            "bbox": (rx, ry, rw, rh),
            "ref_gray": cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY),
        }
        self.since[asset_id] = None

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> Dict[str, Any]:
        alerts = []
        states = {}

        if frame_bgr is None or frame_bgr.size == 0:
            return {"alerts": alerts, "states": states}

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = frame_gray.shape

        for asset_id, data in self.assets.items():
            rx, ry, rw, rh = data["bbox"]
            rx, ry = max(0, min(w - 1, rx)), max(0, min(h - 1, ry))
            rw, rh = max(1, min(w - rx, rw)), max(1, min(h - ry, rh))

            crop = frame_gray[ry : ry + rh, rx : rx + rw]
            ref = data["ref_gray"]

            # Resize if dimensions differ slightly
            if crop.shape != ref.shape:
                crop = cv2.resize(crop, (ref.shape[1], ref.shape[0]))

            diff = float(np.mean(cv2.absdiff(crop, ref)))
            is_different = diff > self.diff_threshold

            if is_different:
                if self.since[asset_id] is None:
                    self.since[asset_id] = timestamp
                elif timestamp - self.since[asset_id] >= self.hold_s:
                    alerts.append({
                        "asset_id": asset_id,
                        "type": "ALARM_THEFT",
                        "timestamp": timestamp,
                        "reason": f"Pixel difference {diff:.1f} > {self.diff_threshold}",
                    })
                    states[asset_id] = "ALARM_THEFT"
                else:
                    states[asset_id] = "SUSPECTED_CHANGE"
            else:
                self.since[asset_id] = None
                states[asset_id] = "SAFE"

        return {"alerts": alerts, "states": states}


class TemporalVotingBaseline:
    """
    Variant 2: Adds N-frame temporal voting & debounce to pixel difference.
    Eliminates transient single-frame spikes, but still triggers false alarms
    when occlusion or lighting changes persist beyond the debounce window.
    """

    def __init__(self, diff_threshold: float = 30.0, required_votes: int = 5):
        self.diff_threshold = diff_threshold
        self.required_votes = required_votes
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.votes: Dict[str, int] = {}

    def reset(self) -> None:
        self.assets.clear()
        self.votes.clear()

    def register_asset(
        self,
        asset_id: str,
        region: List[List[int]],
        baseline_crop: np.ndarray,
        empty_crop: Optional[np.ndarray] = None
    ) -> None:
        pts = np.array(region, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        self.assets[asset_id] = {
            "bbox": (rx, ry, rw, rh),
            "ref_gray": cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY),
        }
        self.votes[asset_id] = 0

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> Dict[str, Any]:
        alerts = []
        states = {}

        if frame_bgr is None or frame_bgr.size == 0:
            return {"alerts": alerts, "states": states}

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = frame_gray.shape

        for asset_id, data in self.assets.items():
            rx, ry, rw, rh = data["bbox"]
            rx, ry = max(0, min(w - 1, rx)), max(0, min(h - 1, ry))
            rw, rh = max(1, min(w - rx, rw)), max(1, min(h - ry, rh))

            crop = frame_gray[ry : ry + rh, rx : rx + rw]
            ref = data["ref_gray"]

            if crop.shape != ref.shape:
                crop = cv2.resize(crop, (ref.shape[1], ref.shape[0]))

            diff = float(np.mean(cv2.absdiff(crop, ref)))
            if diff > self.diff_threshold:
                self.votes[asset_id] += 1
            else:
                self.votes[asset_id] = max(0, self.votes[asset_id] - 1)

            if self.votes[asset_id] >= self.required_votes:
                alerts.append({
                    "asset_id": asset_id,
                    "type": "ALARM_THEFT",
                    "timestamp": timestamp,
                    "reason": f"Sustained pixel difference for {self.votes[asset_id]} frames",
                })
                states[asset_id] = "ALARM_THEFT"
            else:
                states[asset_id] = "SAFE"

        return {"alerts": alerts, "states": states}


class DescriptorVerificationBaseline:
    """
    Variant 3: Adds LabSobelDescriptorV1 identity verification.
    Resists lighting changes through L2-normalized feature representation and
    detects lookalike swaps, but lacks camera-health monitoring and positive emptiness proofs.
    Consequently misidentifies camera blackouts and occlusions as MISSING.
    """

    def __init__(self, sim_threshold: float = 0.70, hold_s: float = 1.5):
        self.descriptor = LabSobelDescriptorV1()
        self.sim_threshold = sim_threshold
        self.hold_s = hold_s
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.since: Dict[str, Optional[float]] = {}

    def reset(self) -> None:
        self.assets.clear()
        self.since.clear()

    def register_asset(
        self,
        asset_id: str,
        region: List[List[int]],
        baseline_crop: np.ndarray,
        empty_crop: Optional[np.ndarray] = None
    ) -> None:
        pts = np.array(region, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        variants = self.descriptor.generate_variants(baseline_crop)
        self.assets[asset_id] = {
            "bbox": (rx, ry, rw, rh),
            "variants": variants,
        }
        self.since[asset_id] = None

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> Dict[str, Any]:
        alerts = []
        states = {}

        if frame_bgr is None or frame_bgr.size == 0:
            return {"alerts": alerts, "states": states}

        h, w = frame_bgr.shape[:2]

        for asset_id, data in self.assets.items():
            rx, ry, rw, rh = data["bbox"]
            rx, ry = max(0, min(w - 1, rx)), max(0, min(h - 1, ry))
            rw, rh = max(1, min(w - rx, rw)), max(1, min(h - ry, rh))

            crop = frame_bgr[ry : ry + rh, rx : rx + rw]
            live_vec = self.descriptor.extract(crop)

            sims = [self.descriptor.cosine_similarity(live_vec, v) for v in data["variants"]]
            top_sim = max(sims) if sims else 0.0

            if top_sim < self.sim_threshold:
                if self.since[asset_id] is None:
                    self.since[asset_id] = timestamp
                elif timestamp - self.since[asset_id] >= self.hold_s:
                    alerts.append({
                        "asset_id": asset_id,
                        "type": "MISSING",
                        "timestamp": timestamp,
                        "reason": f"Descriptor similarity {top_sim:.3f} < {self.sim_threshold}",
                    })
                    states[asset_id] = "MISSING"
                else:
                    states[asset_id] = "SUSPECTED_CHANGE"
            else:
                self.since[asset_id] = None
                states[asset_id] = "SAFE"

        return {"alerts": alerts, "states": states}


class FullTraceEvaluator:
    """
    Variant 4 & 5: Full TRACE Engine.
    Executes the genuine production edge vision and state machine pipeline on raw frames:
    - CameraHealthMonitor: Obstruction, freeze, offline detection
    - FrameAligner: Homography drift correction & tamper knock detection
    - PeopleSegmenter: Human occlusion masking & visibility calculation
    - RegionVerifier: Fused LabSobelDescriptorV1 + SIFT/ORB + Empty ROI SSIM proof
    - AssetFSM: Positive evidence rule, DoubtEngine, temporal confirmation & hysteresis
    - PolicyEngine: Authorized moves & alert deduplication
    """

    def __init__(self):
        self.health_monitor = CameraHealthMonitor(hold_sec=1.5, release_sec=3.0)
        self.aligner = FrameAligner()
        self.people_segmenter = PeopleSegmenter(mask_dilation_px=15, history_len=3)
        self.verifier = RegionVerifier()
        self.relocator = AssetRelocator()
        self.policy_engine = PolicyEngine()
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.fsms: Dict[str, AssetFSM] = {}

    def reset(self) -> None:
        self.health_monitor = CameraHealthMonitor(hold_sec=1.5, release_sec=3.0)
        self.aligner = FrameAligner()
        self.people_segmenter = PeopleSegmenter(mask_dilation_px=15, history_len=3)
        self.relocator = AssetRelocator()
        self.assets.clear()
        self.fsms.clear()

    def register_asset(
        self,
        asset_id: str,
        region: List[List[int]],
        baseline_crop: np.ndarray,
        empty_crop: Optional[np.ndarray] = None,
        class_name: str = "safe",
        criticality: int = 2,
    ) -> None:
        pts = np.array(region, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        variants = self.verifier.extractor.generate_variants(baseline_crop)

        self.assets[asset_id] = {
            "region": region,
            "bbox": (rx, ry, rw, rh),
            "baseline_median": baseline_crop.copy(),
            "variants": variants,
            "empty_roi": empty_crop.copy() if empty_crop is not None else np.full_like(baseline_crop, 160),
            "class_name": class_name,
            "criticality": criticality,
        }

        # Initialize FSM with standard configuration
        cfg = FSMConfig(
            move_frac=0.10,
            present=0.70,
            vis_floor=0.85,
            missing_wait_s=5.0, # Responsive for evaluation protocol
            swap_s=3.0,
            moved_votes=5,
            returned_hold_s=3.0,
            half_life_s=30.0,
        )
        self.fsms[asset_id] = AssetFSM(cfg)

    def calibrate(self, reference_frame: np.ndarray) -> None:
        """Calibrates alignment and camera health baselines from clean reference frame."""
        self.aligner.register_reference(reference_frame)
        self.health_monitor.calibrate_baseline(reference_frame)

    def process_frame(self, frame_bgr: Optional[np.ndarray], timestamp: float) -> Dict[str, Any]:
        alerts = []
        states = {}

        # 1. Camera Health Evaluation
        health_res = self.health_monitor.check(frame_bgr, timestamp, landmark_ratio=self.aligner.inlier_ratio)
        camera_ok = health_res["ok"] and not health_res["is_offline"]

        if not camera_ok or frame_bgr is None:
            # Report camera health alerts if offline or obstructed
            cam_state = "CAMERA_OBSTRUCTED" if health_res.get("is_obstructed") else "SYSTEM_OFFLINE"
            alerts.append({
                "asset_id": "camera",
                "type": cam_state,
                "timestamp": timestamp,
                "reason": health_res.get("reason", "Camera health failure"),
            })
            for asset_id in self.assets:
                # Step FSM with camera_ok=False
                obs = Obs(
                    t=timestamp,
                    present_score=0.0,
                    drift_frac=0.0,
                    visibility=0.0,
                    empty_match=0.0,
                    camera_ok=False,
                )
                fsm_state = self.fsms[asset_id].step(obs).value
                states[asset_id] = fsm_state
            return {"alerts": alerts, "states": states, "camera_ok": False, "camera_status": cam_state}

        # 2. Alignment & Tampering
        align_res = self.aligner.align_frame(frame_bgr)
        is_cam_tampered = align_res.get("is_tampered", False)

        if is_cam_tampered:
            alerts.append({
                "asset_id": "camera",
                "type": "CAMERA_TAMPERED",
                "timestamp": timestamp,
                "reason": f"Displacement {align_res.get('shift_px', 0)}px exceeds threshold",
            })

        # 3. People segmentation
        person_mask = self.people_segmenter.segment(frame_bgr)
        h, w = frame_bgr.shape[:2]

        # 4. Asset verification
        for asset_id, data in self.assets.items():
            fsm = self.fsms[asset_id]
            aligned_region = self.aligner.transform_polygon(data["region"])
            pts = np.array(aligned_region, dtype=np.int32)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            rx, ry = max(0, min(w - 1, rx)), max(0, min(h - 1, ry))
            rw, rh = max(1, min(w - rx, rw)), max(1, min(h - ry, rh))

            live_crop = frame_bgr[ry : ry + rh, rx : rx + rw]

            # Visibility & occlusion
            _, visibility = self.people_segmenter.calculate_visibility_and_occlusion(
                person_mask=person_mask,
                region_pts=aligned_region,
                frame_shape=(h, w)
            )

            # Verification against baseline variants and empty ROI
            ver_res = self.verifier.verify_asset(
                live_crop_bgr=live_crop,
                baseline_variants=data["variants"],
                baseline_median_crop=data["baseline_median"],
                empty_roi_bgr=data["empty_roi"],
            )

            # Drift & relocation calculation
            _, drift_frac = self.verifier.calculate_centroid_drift(aligned_region, data["region"])
            found_elsewhere = False

            if ver_res["present_score"] < 0.70:
                found_elsewhere, found_bbox, _ = self.relocator.search(
                    frame_bgr=frame_bgr,
                    template_crop_bgr=data["baseline_median"],
                    exclude_zone=data["region"],
                    person_mask=person_mask,
                )
                if found_elsewhere and found_bbox is not None:
                    bx, by, bw, bh = found_bbox
                    new_cx, new_cy = bx + bw / 2.0, by + bh / 2.0
                    base_cx, base_cy = rx + rw / 2.0, ry + rh / 2.0
                    drift_px = math.hypot(new_cx - base_cx, new_cy - base_cy)
                    diag = max(1.0, math.hypot(rw, rh))
                    drift_frac = max(drift_frac, drift_px / diag)

            obs = Obs(
                t=timestamp,
                present_score=ver_res["present_score"],
                drift_frac=drift_frac,
                visibility=visibility,
                empty_match=ver_res["empty_match"],
                camera_ok=camera_ok and not is_cam_tampered,
                found_elsewhere=found_elsewhere,
            )

            prev_state = fsm.state.value
            new_state = fsm.step(obs).value
            states[asset_id] = new_state

            # Alert triggering on confirmed transition
            if new_state != prev_state and new_state not in ("SAFE", "OCCLUDED", "RETURNED", "UNKNOWN"):
                alerts.append({
                    "asset_id": asset_id,
                    "type": new_state,
                    "timestamp": timestamp,
                    "confidence": ver_res["present_score"],
                    "reason": f"Confirmed state transition from {prev_state} to {new_state}",
                })

        return {
            "alerts": alerts,
            "states": states,
            "camera_ok": camera_ok and not is_cam_tampered,
            "camera_status": "TAMPERED" if is_cam_tampered else "HEALTHY",
        }
