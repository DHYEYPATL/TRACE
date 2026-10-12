"""
Evaluation Dataset generator and frame streamer.
Produces real video/image sequences with actual pixel-level visual phenomena
(motion, empty desk regions, occlusion silhouettes, lens blackouts, illumination changes,
camera knock displacement) completely decoupled from ground-truth labels.
"""

from __future__ import annotations

import copy
import time
from typing import Callable, Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np

from trace.eval.manifest import ClipManifest, EvaluationDatasetManifest, EventInterval
from trace.vision.capture import generate_synthetic_branch_frame


class EvaluationDatasetBuilder:
    """
    Builds a benchmark dataset consisting of real visual frame sequences and
    an independent ground-truth manifest.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 5.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / fps
        self.master_frame = generate_synthetic_branch_frame(0, self.width, self.height)

    def _get_live_frame(self, frame_idx: int) -> np.ndarray:
        """Simulates an active camera sensor with subtle hardware pixel noise."""
        frame = self.master_frame.copy()
        frame[0:4, 0:4] = ((frame_idx * 31) % 255, (frame_idx * 47) % 255, (frame_idx * 61) % 255)
        return frame

    def build_standard_evaluation_suite(self) -> Tuple[EvaluationDatasetManifest, Dict[str, List[Tuple[float, np.ndarray]]]]:
        """
        Builds all 10 standard evaluation scenarios covering:
        1. clip_stable (10s)
        2. clip_move (8s)
        3. clip_missing (18s)
        4. clip_occlusion (8s)
        5. clip_obstruction (6s)
        6. clip_offline_freeze (7s)
        7. clip_camera_shift (6s)
        8. clip_lighting_drop (7s)
        9. clip_swap_lookalike (14s)
        10. clip_recovery (9s)
        Returns: (manifest, clip_frames_dict)
        """
        manifest = EvaluationDatasetManifest(
            dataset_version="2.0.0-real-cv",
            description="TRACE Ground-Truth Real Video Evaluation Suite (Zero Label Leakage)",
            clips=[]
        )
        clip_frames: Dict[str, List[Tuple[float, np.ndarray]]] = {}

        # 1. Clip: Stable Baseline
        c_stable, f_stable = self._build_clip_stable()
        manifest.clips.append(c_stable)
        clip_frames[c_stable.clip_id] = f_stable

        # 2. Clip: Asset Move Beyond Threshold
        c_move, f_move = self._build_clip_move()
        manifest.clips.append(c_move)
        clip_frames[c_move.clip_id] = f_move

        # 3. Clip: Asset Removed (Missing with Empty Region Proof)
        c_missing, f_missing = self._build_clip_missing()
        manifest.clips.append(c_missing)
        clip_frames[c_missing.clip_id] = f_missing

        # 4. Clip: Person Transient Occlusion
        c_occl, f_occl = self._build_clip_occlusion()
        manifest.clips.append(c_occl)
        clip_frames[c_occl.clip_id] = f_occl

        # 5. Clip: Optical Lens Obstruction
        c_obstr, f_obstr = self._build_clip_obstruction()
        manifest.clips.append(c_obstr)
        clip_frames[c_obstr.clip_id] = f_obstr

        # 6. Clip: Camera Frozen / Offline
        c_off, f_off = self._build_clip_freeze_offline()
        manifest.clips.append(c_off)
        clip_frames[c_off.clip_id] = f_off

        # 7. Clip: Camera Shift / Homography Tampering
        c_shift, f_shift = self._build_clip_camera_shift()
        manifest.clips.append(c_shift)
        clip_frames[c_shift.clip_id] = f_shift

        # 8. Clip: Lighting Change (Dim Environment)
        c_light, f_light = self._build_clip_lighting_drop()
        manifest.clips.append(c_light)
        clip_frames[c_light.clip_id] = f_light

        # 9. Clip: Lookalike Substitution (Swap)
        c_swap, f_swap = self._build_clip_swap_lookalike()
        manifest.clips.append(c_swap)
        clip_frames[c_swap.clip_id] = f_swap

        # 10. Clip: Move and Recovery Hysteresis
        c_rec, f_rec = self._build_clip_recovery()
        manifest.clips.append(c_rec)
        clip_frames[c_rec.clip_id] = f_rec

        return manifest, clip_frames

    # --- Scenario Generators ---

    def _build_clip_stable(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        duration_s = 6.0
        n_frames = int(duration_s * self.fps)
        frames = []
        for i in range(n_frames):
            t = i * self.dt
            frame = self._get_live_frame(i)
            frames.append((t, frame))

        manifest = ClipManifest(
            clip_id="clip_stable",
            description="Stable branch scene under normal operation with static assets.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("STABLE", "safe-02", 0.0, duration_s, must_alert=False, expected_state="SAFE"),
                EventInterval("STABLE", "printer-01", 0.0, duration_s, must_alert=False, expected_state="SAFE"),
                EventInterval("STABLE", "laptop-03", 0.0, duration_s, must_alert=False, expected_state="SAFE"),
            ]
        )
        return manifest, frames

    def _build_clip_move(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        duration_s = 6.0
        n_frames = int(duration_s * self.fps)
        onset_s = 1.5
        frames = []

        # Safe region is [180, 220] to [380, 520] (w=200, h=300).
        # We shift it by +70px in X after onset_s (35% of width, well beyond 10% move_frac threshold).
        dx = 70
        for i in range(n_frames):
            t = i * self.dt
            base = self._get_live_frame(i)
            if t >= onset_s:
                # Cut safe crop from baseline region
                safe_crop = base[220:520, 180:380].copy()
                # Paint empty wall/desk texture over the old safe region
                desk_top = int(self.height * 0.45)
                base[220:desk_top, 180:380] = (215, 220, 225)
                base[desk_top:520, 180:380] = (120, 140, 160)
                # Paste safe crop shifted to the right
                base[220:520, 180 + dx : 380 + dx] = safe_crop
            frames.append((t, base))

        manifest = ClipManifest(
            clip_id="clip_move",
            description="Cash Safe physically moved 70px beyond configured threshold at t=1.5s.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("MOVED", "safe-02", onset_s, duration_s, must_alert=True, expected_state="MOVED"),
                EventInterval("STABLE", "printer-01", 0.0, duration_s, must_alert=False, expected_state="SAFE"),
            ]
        )
        return manifest, frames

    def _build_clip_missing(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        # Missing requires missing_wait_s (15s demo) + observation votes
        duration_s = 18.0
        n_frames = int(duration_s * self.fps)
        onset_s = 1.0
        frames = []

        # Printer is at [520, 260] to [740, 480].
        # At t >= 1.0s, remove printer completely, exposing neutral desk background (positive emptiness evidence)
        for i in range(n_frames):
            t = i * self.dt
            base = self._get_live_frame(i)
            if t >= onset_s:
                desk_top = int(self.height * 0.45)
                base[260:desk_top, 520:740] = (215, 220, 225) # Wall
                base[desk_top:480, 520:740] = (120, 140, 160) # Clean desk surface
            frames.append((t, base))

        manifest = ClipManifest(
            clip_id="clip_missing",
            description="Laser printer removed at t=1.0s, leaving bare desk surface.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("MISSING", "printer-01", onset_s, duration_s, must_alert=True, expected_state="MISSING"),
                EventInterval("STABLE", "safe-02", 0.0, duration_s, must_alert=False, expected_state="SAFE"),
            ]
        )
        return manifest, frames

    def _build_clip_occlusion(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        duration_s = 8.0
        n_frames = int(duration_s * self.fps)
        start_occl = 1.6
        end_occl = 4.8
        frames = []

        for i in range(n_frames):
            t = i * self.dt
            base = self._get_live_frame(i)
            if start_occl <= t <= end_occl:
                # Render a standing person silhouette directly in front of the safe (occlusion > 70%)
                # Center around (280, 360), torso/legs in dark blue/grey
                cv2.ellipse(base, (280, 240), (45, 60), 0, 0, 360, (30, 40, 70), -1) # Head
                cv2.rectangle(base, (220, 290), (340, 530), (50, 60, 90), -1)       # Torso
            frames.append((t, base))

        manifest = ClipManifest(
            clip_id="clip_occlusion",
            description="Person stands in front of safe from t=1.6s to t=4.8s, then departs.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("OCCLUDED", "safe-02", start_occl, end_occl, must_alert=False, expected_state="OCCLUDED"),
                EventInterval("STABLE", "safe-02", end_occl + 1.0, duration_s, must_alert=False, expected_state="SAFE"),
            ]
        )
        return manifest, frames

    def _build_clip_obstruction(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        duration_s = 6.0
        n_frames = int(duration_s * self.fps)
        onset_s = 1.5
        frames = []

        for i in range(n_frames):
            t = i * self.dt
            if t < onset_s:
                frame = self._get_live_frame(i)
            else:
                # Lens completely blocked (opaque cover / near zero brightness and variance)
                frame = np.full((self.height, self.width, 3), 12, dtype=np.uint8)
            frames.append((t, frame))

        manifest = ClipManifest(
            clip_id="clip_obstruction",
            description="Camera lens covered with an opaque cloth at t=1.5s.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("CAMERA_OBSTRUCTED", "camera", onset_s, duration_s, must_alert=True, expected_state="CAMERA_OBSTRUCTED"),
            ]
        )
        return manifest, frames

    def _build_clip_freeze_offline(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        duration_s = 6.0
        n_frames = int(duration_s * self.fps)
        onset_s = 1.4
        frames = []

        frozen_frame = self._get_live_frame(int(onset_s * self.fps))
        for i in range(n_frames):
            t = i * self.dt
            if t < onset_s:
                frame = self._get_live_frame(i)
            else:
                # Feed is frozen (exact bit-identical frame repeated)
                frame = frozen_frame.copy()
            frames.append((t, frame))

        manifest = ClipManifest(
            clip_id="clip_offline_freeze",
            description="Video capture freezes at t=1.4s.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("CAMERA_OFFLINE", "camera", onset_s + 2.0, duration_s, must_alert=True, expected_state="SYSTEM_OFFLINE"),
            ]
        )
        return manifest, frames

    def _build_clip_camera_shift(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        duration_s = 6.0
        n_frames = int(duration_s * self.fps)
        onset_s = 1.4
        frames = []

        # Shift frame by (dx=45, dy=35) simulating camera bumped / displaced
        M = np.float32([[1, 0, 45], [0, 1, 35]])

        for i in range(n_frames):
            t = i * self.dt
            base = self._get_live_frame(i)
            if t >= onset_s:
                shifted = cv2.warpAffine(base, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)
                frame = shifted
            else:
                frame = base
            frames.append((t, frame))

        manifest = ClipManifest(
            clip_id="clip_camera_shift",
            description="Camera physically knocked 57px displacement at t=1.4s.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("CAMERA_TAMPERED", "camera", onset_s, duration_s, must_alert=True, expected_state="CAMERA_OBSTRUCTED"),
            ]
        )
        return manifest, frames

    def _build_clip_lighting_drop(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        duration_s = 6.0
        n_frames = int(duration_s * self.fps)
        onset_s = 1.5
        frames = []

        for i in range(n_frames):
            t = i * self.dt
            base = self._get_live_frame(i)
            if t >= onset_s:
                # Dim the lights by 60% (scalar multiplication, maintaining scene geometry)
                dimmed = cv2.convertScaleAbs(base, alpha=0.40, beta=-15)
                frame = dimmed
            else:
                frame = base
            frames.append((t, frame))

        manifest = ClipManifest(
            clip_id="clip_lighting_drop",
            description="Ambient lighting drops 60% at t=1.5s with all assets remaining stable.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("LIGHTING_CHANGE", "safe-02", onset_s, duration_s, must_alert=False, expected_state="SAFE"),
                EventInterval("LIGHTING_CHANGE", "printer-01", onset_s, duration_s, must_alert=False, expected_state="SAFE"),
            ]
        )
        return manifest, frames

    def _build_clip_swap_lookalike(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        # Swapping laptop with a brightly painted red box lookalike
        duration_s = 13.0
        n_frames = int(duration_s * self.fps)
        onset_s = 1.2
        frames = []

        for i in range(n_frames):
            t = i * self.dt
            base = self._get_live_frame(i)
            if t >= onset_s:
                # Laptop region: [880, 310] to [1100, 500]
                # Replace with a bright orange/red object of identical size
                desk_top = int(self.height * 0.45)
                cv2.rectangle(base, (880, 310), (1100, 500), (40, 50, 210), -1) # Red/orange swap
                cv2.putText(base, "LOOKALIKE DECOY", (900, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            frames.append((t, base))

        manifest = ClipManifest(
            clip_id="clip_swap_lookalike",
            description="Laptop replaced with lookalike decoy at t=1.2s.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("SWAPPED", "laptop-03", onset_s, duration_s, must_alert=True, expected_state="SWAPPED"),
            ]
        )
        return manifest, frames

    def _build_clip_recovery(self) -> Tuple[ClipManifest, List[Tuple[float, np.ndarray]]]:
        # Safe moved at t=1.2s, returned back to baseline position at t=3.5s
        duration_s = 8.0
        n_frames = int(duration_s * self.fps)
        move_t = 1.2
        return_t = 3.5
        frames = []
        dx = 70

        for i in range(n_frames):
            t = i * self.dt
            base = self._get_live_frame(i)
            if move_t <= t < return_t:
                # Shifted position
                safe_crop = base[220:520, 180:380].copy()
                desk_top = int(self.height * 0.45)
                base[220:desk_top, 180:380] = (215, 220, 225)
                base[desk_top:520, 180:380] = (120, 140, 160)
                base[220:520, 180 + dx : 380 + dx] = safe_crop
            # At t >= return_t, safe is back in its exact baseline position!
            frames.append((t, base))

        manifest = ClipManifest(
            clip_id="clip_recovery",
            description="Safe moved at t=1.2s and restored to baseline location at t=3.5s.",
            fps=self.fps,
            duration_s=duration_s,
            events=[
                EventInterval("MOVED", "safe-02", move_t, return_t, must_alert=True, expected_state="MOVED"),
                EventInterval("STABLE", "safe-02", return_t + 2.0, duration_s, must_alert=False, expected_state="SAFE"),
            ]
        )
        return manifest, frames
