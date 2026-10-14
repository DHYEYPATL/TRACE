"""Concurrent camera, detector, object-registration, and alert service."""
from __future__ import annotations

import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .identity import IdentityCache, GALLERY_MAX
from .storage import Store
from .vision import VisionEngine, _sharpness

# ------------------------------------------------------------------
# Monitoring mode helpers
# ------------------------------------------------------------------

_MODE         = os.getenv("ASSETWATCH_MODE",      "always")   # always|day|office|night
_DAY_START    = os.getenv("ASSETWATCH_DAY_START", "08:00")
_DAY_END      = os.getenv("ASSETWATCH_DAY_END",   "20:00")
_IDLE_FPS     = max(0.5, float(os.getenv("ASSETWATCH_IDLE_FPS", "2.0")))


def _parse_hhmm(s: str) -> tuple[int, int]:
    try:
        h, m = s.strip().split(":")
        return int(h), int(m)
    except Exception:
        return 8, 0


def _is_active_hours() -> bool:
    now = datetime.now()
    sh, sm = _parse_hhmm(_DAY_START)
    eh, em = _parse_hhmm(_DAY_END)
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end   = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now < end


def _mode_is_idle() -> bool:
    """Return True when the current mode says inference should run at idle FPS."""
    if _MODE == "always":
        return False
    if _MODE == "night":
        return True
    if _MODE == "day":
        return not _is_active_hours()
    # "office" — active during day hours; night is idle.
    if _MODE == "office":
        return not _is_active_hours()
    return False


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter   = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a  = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b  = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _parse_iso_age(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())
    except (ValueError, TypeError):
        return 0.0


def _save_crop_thumbnail(path: Path, crop: np.ndarray,
                          max_side: int = 160, quality: int = 75) -> None:
    """Save a compact JPEG thumbnail (max 160 px on longest side, Q=75)."""
    h, w = crop.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        crop  = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, quality])


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------

class CameraService:
    def __init__(self, store: Store, data_dir: str | Path,
                 camera_index: int | None = None, target_infer_fps: float | None = None,
                 missing_after_s: float | None = None, movement_fraction: float | None = None,
                 model_name: str | None = None):
        self.store         = store
        self.data_dir      = Path(data_dir)
        self.reference_dir = self.data_dir / "references"
        self.reference_dir.mkdir(parents=True, exist_ok=True)

        self.camera_index      = int(os.getenv("ASSETWATCH_CAMERA_INDEX", str(camera_index if camera_index is not None else 1)))
        self.target_infer_fps  = max(1.0, float(os.getenv("ASSETWATCH_INFER_FPS",         str(target_infer_fps or 8.0))))
        self.missing_after_s   = max(5.0, float(os.getenv("ASSETWATCH_MISSING_AFTER",      str(missing_after_s or 25.0))))
        self.movement_fraction = float(os.getenv("ASSETWATCH_MOVEMENT_FRACTION",           str(movement_fraction or 0.11)))
        self.model_name        = model_name or os.getenv("ASSETWATCH_MODEL", "yolo11n.pt")
        self._identity_ttl_s   = self.missing_after_s + 10.0

        self.engine          = VisionEngine(
            self.model_name,
            imgsz=int(os.getenv("ASSETWATCH_IMGSZ",       "640")),
            conf =float(os.getenv("ASSETWATCH_CONFIDENCE", "0.25")),
        )
        self._identity_cache = IdentityCache()

        self._lock  = threading.RLock()
        self._stop  = threading.Event()
        self._threads: list[threading.Thread] = []

        # Camera state.
        self.latest_frame:   np.ndarray | None = None
        self.frame_id        = 0
        self.frame_timestamp = 0.0
        self.frame_width     = 0
        self.frame_height    = 0
        self.camera_open     = False
        self.camera_error:   str | None = None
        self.capture_fps     = 0.0

        # Inference state.
        self.infer_fps            = 0.0
        self.infer_latency_ms     = 0.0
        self.last_infer_timestamp = 0.0
        self.detections:  list[dict[str, Any]] = []
        self.selected:    dict[str, Any] | None = None
        self.device_status  = "Loading on first inference"
        self.startup_error: str | None = None

        # Track → asset short-term cache.
        self.track_assignments:     dict[int, dict[str, Any]] = {}
        self.custom_trackers:       dict[str, dict[str, Any]] = {}
        self.track_feature_checked: dict[int, float] = {}

        # Per-asset monotonic timestamps.
        self.asset_last_seen_mono: dict[str, float] = {}
        self.asset_visible_boxes:  dict[str, list[float]] = {}

        # Misc.
        self._last_flush_mono     = time.monotonic()
        self._last_enrich_check   = 0.0    # throttle enrichment checks
        self._person_last_seen    = 0.0    # for office mode

        self._init_asset_times()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _init_asset_times(self) -> None:
        now = time.monotonic()
        for asset in self.store.list_assets():
            age = _parse_iso_age(asset.get("last_seen"))
            self.asset_last_seen_mono[asset["asset_id"]] = now - age
            s = self._identity_cache.ensure(asset["asset_id"])
            s.last_seen_mono = now - age
            # Sync gallery size from stored profile so the coverage meter is accurate on startup.
            s.gallery_size   = len(asset.get("profile", {}).get("samples", []))

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._capture_loop,   name="assetwatch-camera",    daemon=True),
            threading.Thread(target=self._inference_loop, name="assetwatch-inference", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._flush_rolling_embeddings()
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()

    def change_camera(self, index: int) -> None:
        if index < 0 or index > 9:
            raise ValueError("Camera index must be between 0 and 9")
        with self._lock:
            self.camera_index = int(index)
            self.camera_open  = False
            self.camera_error = f"Switching to camera {index}…"
        self._requested_camera_index = int(index)

    # ------------------------------------------------------------------
    # Camera loop
    # ------------------------------------------------------------------

    def _open_camera(self, index: int):
        attempts = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for backend in attempts:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(os.getenv("ASSETWATCH_CAMERA_WIDTH",  "1280")))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.getenv("ASSETWATCH_CAMERA_HEIGHT", "720")))
                cap.set(cv2.CAP_PROP_FPS, 30)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, frame
            cap.release()
        return None, None

    def _capture_loop(self) -> None:
        cap          = None
        active_index = None
        last_stamp   = time.monotonic()
        while not self._stop.is_set():
            requested = getattr(self, "_requested_camera_index", self.camera_index)
            if cap is None or not cap.isOpened() or requested != active_index:
                if cap is not None:
                    cap.release()
                active_index = requested
                cap, first   = self._open_camera(active_index)
                if cap is None:
                    with self._lock:
                        self.camera_open  = False
                        self.camera_error = f"Cannot open camera index {active_index}. Check USB connection and try another index."
                    self._stop.wait(1.5)
                    continue
                with self._lock:
                    self.camera_open     = True
                    self.camera_error    = None
                    self.latest_frame    = first
                    self.frame_height, self.frame_width = first.shape[:2]
                    self.frame_timestamp = time.monotonic()
                    self.frame_id       += 1
                last_stamp = time.monotonic()
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                with self._lock:
                    self.camera_open  = False
                    self.camera_error = "Camera stopped returning frames; reconnecting…"
                cap.release()
                cap = None
                self._stop.wait(0.5)
                continue
            dt = now - last_stamp
            last_stamp = now
            if dt > 0:
                measured = min(120.0, 1.0 / dt)
                with self._lock:
                    self.capture_fps     = measured if self.capture_fps <= 0 else 0.85 * self.capture_fps + 0.15 * measured
                    self.camera_open     = True
                    self.camera_error    = None
                    self.latest_frame    = frame
                    self.frame_height, self.frame_width = frame.shape[:2]
                    self.frame_timestamp = now
                    self.frame_id       += 1
        if cap is not None:
            cap.release()
        with self._lock:
            self.camera_open = False

    # ------------------------------------------------------------------
    # Inference loop (monitoring-mode aware)
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        last_processed_id = -1
        last_infer_mono   = 0.0

        while not self._stop.is_set():
            idle       = _mode_is_idle()
            min_period = 1.0 / (_IDLE_FPS if idle else self.target_infer_fps)

            with self._lock:
                frame    = None if self.latest_frame is None else self.latest_frame.copy()
                frame_id = self.frame_id
                fresh    = self.camera_open and (time.monotonic() - self.frame_timestamp < 3.0)
            if frame is None or frame_id == last_processed_id:
                self._update_missing_states(camera_healthy=fresh)
                self._stop.wait(0.02)
                continue
            now = time.monotonic()
            if now - last_infer_mono < min_period and last_processed_id >= 0:
                self._stop.wait(max(0.005, min_period - (now - last_infer_mono)))
                continue

            last_processed_id = frame_id
            start_wait  = time.monotonic()
            infer_start = time.perf_counter()
            try:
                if not self.engine.ready:
                    self.engine.initialize()
                    self.device_status = self.engine.device_name
                    with self._lock:
                        if self.latest_frame is not None:
                            frame             = self.latest_frame.copy()
                            frame_id          = self.frame_id
                            last_processed_id = frame_id

                detections = self.engine.detect(frame)
                self.device_status = self.engine.device_name
                self.startup_error = None

                # In idle mode skip heavy matching; just keep camera alive.
                if not idle:
                    detections = self._associate_assets(frame, detections)
                    detections = self._update_custom_trackers(frame, detections)

                elapsed = (time.perf_counter() - infer_start) * 1000.0
                with self._lock:
                    self.detections       = detections
                    self.infer_latency_ms = elapsed
                    now_infer = time.monotonic()
                    interval  = now_infer - self.last_infer_timestamp if self.last_infer_timestamp else 0.0
                    if interval > 0:
                        mfps = min(60.0, 1.0 / interval)
                        self.infer_fps = mfps if self.infer_fps <= 0 else 0.75 * self.infer_fps + 0.25 * mfps
                    self.last_infer_timestamp = now_infer
                last_infer_mono = time.monotonic()

                with self._lock:
                    healthy_now = self.camera_open and (time.monotonic() - self.frame_timestamp < 3.0)
                    fw, fh = self.frame_width or 1280, self.frame_height or 720
                self._update_asset_states(detections, fw, fh, camera_healthy=healthy_now)

                # Periodic flush (every 30 s).
                if time.monotonic() - self._last_flush_mono > 30.0:
                    self._flush_rolling_embeddings()
                    self._last_flush_mono = time.monotonic()

            except Exception as exc:
                self.startup_error = f"{type(exc).__name__}: {exc}"
                self.device_status = "Model unavailable"
                self._stop.wait(0.8)

            elapsed_loop = time.monotonic() - start_wait
            if elapsed_loop < 0.003:
                self._stop.wait(0.003)

    # ------------------------------------------------------------------
    # Asset association — two-stage identity matching + enrichment
    # ------------------------------------------------------------------

    def _associate_assets(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        assets = self.store.list_assets()
        fw, fh = frame.shape[1], frame.shape[0]
        now    = time.monotonic()

        visible_track_ids = {int(d["track_id"]) for d in detections if d.get("track_id") is not None}

        # Maintain short-term track cache.
        claimed: dict[str, int] = {}
        for tid, assignment in list(self.track_assignments.items()):
            if tid in visible_track_ids and assignment.get("asset_id"):
                claimed[assignment["asset_id"]] = tid
            elif assignment.get("lost_since") is None:
                assignment["lost_since"] = now
            elif now - assignment["lost_since"] > 8:
                self.track_assignments.pop(tid, None)
                self.track_feature_checked.pop(tid, None)

        for tid, checked_at in list(self.track_feature_checked.items()):
            if tid not in visible_track_ids and now - checked_at > 20:
                self.track_feature_checked.pop(tid, None)

        identity_hints = {s.asset_id: s for s in self._identity_cache.all_states()}

        for det in detections:
            tid = det.get("track_id")

            # --- Fast path: existing track assignment re-verified every 3 s ---
            if tid is not None and tid in self.track_assignments and self.track_assignments[tid].get("asset_id"):
                assignment = self.track_assignments[tid]
                assignment["lost_since"] = None
                if now - self.track_feature_checked.get(tid, 0.0) < 3.0:
                    asset = self.store.get_asset(assignment["asset_id"])
                    if asset:
                        det["asset_id"]   = asset["asset_id"]
                        det["asset_name"] = asset["name"]
                        det["similarity"] = assignment.get("similarity")
                        self._identity_cache.on_matched(
                            asset["asset_id"], tid, det["bbox"], [], assignment.get("similarity") or 0.0)
                        continue
                # Re-verification.
                self.track_feature_checked[tid] = now
                asset = self.store.get_asset(assignment["asset_id"])
                crop  = self._crop(frame, det["bbox"])
                try:
                    features       = self.engine.extract_features(crop)
                    gallery        = (asset or {}).get("profile", {}).get("samples", [])
                    assigned_score = max((self.engine.compare_features(features, s) for s in gallery), default=0.0)
                except Exception:
                    assigned_score = 0.0
                    features       = {}
                if asset and assigned_score >= 0.62:
                    assignment["similarity"] = assigned_score
                    det["asset_id"]   = asset["asset_id"]
                    det["asset_name"] = asset["name"]
                    det["similarity"] = assigned_score
                    self._identity_cache.on_matched(
                        asset["asset_id"], tid, det["bbox"],
                        features.get("embedding", []), assigned_score)
                    # Enrichment check on re-verify.
                    self._maybe_enrich(asset, features, assigned_score, det["bbox"])
                    continue
                self.track_assignments.pop(tid, None)
                claimed.pop(assignment.get("asset_id"), None)

            # --- Cooldown: skip very recently checked tracks ---
            if tid is not None and now - self.track_feature_checked.get(tid, 0.0) < 2.0:
                continue
            if det["source"] != "yolo":
                continue

            self.track_feature_checked[tid if tid is not None else -1] = now
            crop = self._crop(frame, det["bbox"])
            if crop.size == 0 or crop.shape[0] < 18 or crop.shape[1] < 18:
                continue

            detected_class = det["class_name"].lower()
            compatible     = [a for a in assets if
                              a.get("category", "custom").lower() in ("custom", "any", detected_class)]
            excluded       = {aid for aid, assigned_tid in claimed.items() if assigned_tid != tid}

            try:
                features = self.engine.extract_features(crop)
                aid, score = self.engine.best_asset_match(
                    features, compatible,
                    excluded_asset_ids=excluded,
                    min_score =float(os.getenv("ASSETWATCH_MATCH_THRESHOLD", "0.76")),
                    min_margin=float(os.getenv("ASSETWATCH_MATCH_MARGIN",    "0.07")),
                    identity_hints=identity_hints,
                    candidate_box =det["bbox"],
                    frame_w=fw, frame_h=fh,
                )
            except Exception:
                continue

            if aid:
                asset = self.store.get_asset(aid)
                if not asset:
                    continue
                det["asset_id"]   = aid
                det["asset_name"] = asset["name"]
                det["similarity"] = score

                was_lost = self._identity_cache.is_lost(aid)
                self._identity_cache.on_recovered(
                    aid, tid, det["bbox"], features.get("embedding", []),
                    score or 0.0, from_gallery=True)
                if was_lost:
                    self.store.add_event(aid, "RECOVERED", f"{asset['name']} re-identified after being lost.")
                    det["recovered"] = True

                if tid is not None:
                    self.track_assignments[tid] = {"asset_id": aid, "similarity": score, "lost_since": None}
                claimed[aid] = tid if tid is not None else -1

                # Enrichment check on new match.
                self._maybe_enrich(asset, features, score or 0.0, det["bbox"])

        # Carry forward cached identity for cooldown tracks.
        for det in detections:
            tid = det.get("track_id")
            if tid is not None and tid in self.track_assignments and det.get("asset_id") is None:
                assignment = self.track_assignments[tid]
                asset      = self.store.get_asset(assignment.get("asset_id", ""))
                if asset:
                    det["asset_id"]   = asset["asset_id"]
                    det["asset_name"] = asset["name"]
                    det["similarity"] = assignment.get("similarity")
        return detections

    # ------------------------------------------------------------------
    # Online gallery enrichment
    # ------------------------------------------------------------------

    def _maybe_enrich(self, asset: dict, features: dict,
                       score: float, box: list[float]) -> None:
        """Add a new diverse view to the gallery if quality gates pass."""
        aid     = asset["asset_id"]
        gallery = asset.get("profile", {}).get("samples", [])
        if not self._identity_cache.check_enrich(aid, features.get("embedding", []),
                                                  gallery, score, box):
            return
        emb = self._identity_cache.pop_pending_enrich(aid)
        if emb is None:
            return
        # Reload freshest profile (another thread may have updated it).
        fresh_asset = self.store.get_asset(aid)
        if not fresh_asset:
            return
        profile = fresh_asset["profile"]
        samples = profile.get("samples", [])
        new_sample = {
            "embedding":  emb,
            "histogram":  features.get("histogram", []),
            "orb_b64":    features.get("orb_b64",   ""),
            "orb_rows":   features.get("orb_rows",   0),
            "sharpness":  features.get("sharpness",  0.0),
            "auto":       True,   # mark as auto-enriched
        }
        if len(samples) < GALLERY_MAX:
            samples.append(new_sample)
        else:
            # Replace the sample with the lowest sharpness score.
            worst_idx = min(range(len(samples)),
                            key=lambda i: samples[i].get("sharpness", 0.0))
            samples[worst_idx] = new_sample
        profile["samples"] = samples
        try:
            self.store.update_profile(aid, profile, silent=True)
            s = self._identity_cache.ensure(aid)
            s.gallery_size = len(samples)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Flush rolling embeddings to SQLite
    # ------------------------------------------------------------------

    def _flush_rolling_embeddings(self) -> None:
        for s in self._identity_cache.all_states():
            if not s.rolling_embeddings:
                continue
            asset = self.store.get_asset(s.asset_id)
            if not asset:
                continue
            profile    = asset["profile"]
            latest_emb = list(s.rolling_embeddings)[-1]
            if not isinstance(latest_emb, list) or not latest_emb:
                continue
            new_e  = np.asarray(latest_emb, dtype=np.float32)
            is_dup = False
            for sample in profile.get("samples", []):
                se = np.asarray(sample.get("embedding", []), dtype=np.float32)
                if se.shape == new_e.shape and se.size > 0:
                    cos = float(np.dot(se, new_e) / (np.linalg.norm(se) * np.linalg.norm(new_e) + 1e-9))
                    if cos > 0.98:
                        is_dup = True
                        break
            if not is_dup:
                samples = profile.get("samples", [])
                samples.append({"embedding": latest_emb, "histogram": [],
                                "orb_b64": "", "orb_rows": 0, "auto": True})
                profile["samples"] = samples[-GALLERY_MAX:]
                try:
                    self.store.update_profile(s.asset_id, profile, silent=True)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # CSRT manual trackers
    # ------------------------------------------------------------------

    def _new_csrt(self):
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        raise RuntimeError("CSRT tracker is unavailable. Reinstall opencv-contrib-python.")

    def _update_custom_trackers(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        with self._lock:
            trackers = list(self.custom_trackers.items())
        detected_asset_ids = {d.get("asset_id") for d in detections if d.get("asset_id")}
        for asset_id, entry in trackers:
            try:
                ok, box = entry["tracker"].update(frame)
            except Exception:
                ok, box = False, None
            if ok and box is not None:
                x, y, w, h = [float(v) for v in box]
                bbox = [max(0, x), max(0, y),
                        min(frame.shape[1], x + w), min(frame.shape[0], y + h)]
                if bbox[2] - bbox[0] >= 10 and bbox[3] - bbox[1] >= 10:
                    det = {"key": f"csrt:{asset_id}", "track_id": None,
                           "bbox": [int(v) for v in bbox],
                           "class_name": "manually tracked", "confidence": None,
                           "source": "csrt", "asset_id": asset_id,
                           "asset_name": entry["name"], "similarity": None}
                    if asset_id not in detected_asset_ids:
                        detections.append(det)
                    entry["last_box"] = det["bbox"]
                    entry["failures"] = 0
                    self._try_promote_csrt_to_yolo(asset_id, det["bbox"], detections, frame)
                    continue
            if asset_id in detected_asset_ids:
                entry["failures"] = 0
                continue
            entry["failures"] = int(entry.get("failures", 0)) + 1
            if entry["failures"] >= 2:
                with self._lock:
                    self.custom_trackers.pop(asset_id, None)
                self.store.add_event(asset_id, "TRACKER_LOST",
                                     "Visual tracker lost the item; monitoring will wait for re-detection or manual recapture.")
        return detections

    def _try_promote_csrt_to_yolo(self, asset_id: str, csrt_box: list[float],
                                   detections: list[dict], frame: np.ndarray) -> None:
        asset = self.store.get_asset(asset_id)
        if not asset:
            return
        for det in detections:
            if det.get("source") != "yolo" or det.get("asset_id") is not None:
                continue
            if _box_iou(csrt_box, det["bbox"]) < 0.30:
                continue
            crop = self._crop(frame, det["bbox"])
            if crop.size == 0 or crop.shape[0] < 18 or crop.shape[1] < 18:
                continue
            try:
                features = self.engine.extract_features(crop)
                score    = max((self.engine.compare_features(features, s)
                                for s in asset.get("profile", {}).get("samples", [])), default=0.0)
            except Exception:
                continue
            if score >= 0.70:
                det["asset_id"]   = asset_id
                det["asset_name"] = asset["name"]
                det["similarity"] = score
                tid = det.get("track_id")
                if tid is not None:
                    self.track_assignments[tid] = {"asset_id": asset_id, "similarity": score, "lost_since": None}
                with self._lock:
                    self.custom_trackers.pop(asset_id, None)
                self.store.add_event(asset_id, "PROMOTED",
                                     f"{asset['name']} promoted from manual tracker to AI detection.")
                break

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _update_asset_states(self, detections: list[dict], width: int, height: int,
                              camera_healthy: bool) -> None:
        now     = time.monotonic()
        visible = {det["asset_id"]: det for det in detections if det.get("asset_id")}

        for asset in self.store.list_assets():
            aid = asset["asset_id"]
            if not camera_healthy:
                continue

            if aid in visible:
                det  = visible[aid]
                box  = det["bbox"]
                sim  = det.get("similarity")
                self.asset_last_seen_mono[aid] = now
                self.asset_visible_boxes[aid]  = box
                self.store.observe(aid, box, sim)

                anchor     = asset.get("anchor_box") or box
                frame_diag = max(1.0, (width * width + height * height) ** 0.5)
                acx, acy   = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
                cx,  cy    = (box[0]    + box[2])    / 2, (box[1]    + box[3])    / 2
                displacement = ((cx - acx) ** 2 + (cy - acy) ** 2) ** 0.5 / frame_diag
                threshold    = self.movement_fraction * (0.78 if asset.get("status") == "MOVED" else 1.0)
                high_conf    = sim is None or sim >= 0.70
                moved  = displacement > threshold and high_conf
                status = "MOVED" if moved else "PRESENT"
                msg    = (f"{asset['name']} moved from its registered position" if moved
                          else f"{asset['name']} is visible again")
                self.store.set_status(aid, status, msg if asset.get("status") != status else None)
                self._identity_cache.on_matched(aid, None, box, [], sim or 0.0)
            else:
                age = now - self.asset_last_seen_mono.get(aid, now)
                self._identity_cache.on_lost(aid)
                if age >= self.missing_after_s:
                    self.store.set_status(
                        aid, "MISSING",
                        f"{asset['name']} has not been observed for {int(age)} s. "
                        f"Check the scene before treating this as a confirmed removal.")
                elif age >= 3:
                    self.store.set_status(aid, "CHECKING", None)

        # Hard-evict stale identity states.
        for s in self._identity_cache.all_states():
            if s.lost_since is not None and (now - s.lost_since) > self._identity_ttl_s:
                if self.store.get_asset(s.asset_id) is None:
                    self._identity_cache.remove(s.asset_id)

    def _update_missing_states(self, camera_healthy: bool) -> None:
        if not camera_healthy:
            return
        with self._lock:
            fw = self.frame_width  or 1280
            fh = self.frame_height or 720
        self._update_asset_states([], fw, fh, camera_healthy=True)

    # ------------------------------------------------------------------
    # Selection / registration
    # ------------------------------------------------------------------

    def _crop(self, frame: np.ndarray, box: list[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
        return frame[y1:y2, x1:x2].copy()

    def select_region(self, payload: dict) -> dict:
        with self._lock:
            if self.latest_frame is None:
                raise RuntimeError("Camera has no frame yet")
            frame      = self.latest_frame.copy()
            detections = [dict(d) for d in self.detections]
        h, w = frame.shape[:2]
        if all(k in payload for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = [int(round(float(payload[k]))) for k in ("x1", "y1", "x2", "y2")]
            x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
            if x2 - x1 < 15 or y2 - y1 < 15:
                raise ValueError("Draw a larger box around the object")
            roi      = [x1, y1, x2, y2]
            matching = max(detections, key=lambda d: _box_iou(roi, d["bbox"]), default=None)
            if matching and _box_iou(roi, matching["bbox"]) > 0.30:
                selection = {"mode": "detector", "bbox": matching["bbox"],
                             "track_id": matching.get("track_id"),
                             "class_name": matching["class_name"], "frame_id": self.frame_id}
            else:
                selection = {"mode": "csrt", "bbox": roi, "track_id": None,
                             "class_name": "custom", "frame_id": self.frame_id}
        else:
            x, y     = int(payload.get("x", -1)), int(payload.get("y", -1))
            matching = next((d for d in reversed(detections)
                             if d["bbox"][0] <= x <= d["bbox"][2] and d["bbox"][1] <= y <= d["bbox"][3]), None)
            if not matching:
                raise ValueError("Click inside a detection box, or click-drag to draw a box around any object")
            selection = {"mode": "detector", "bbox": matching["bbox"],
                         "track_id": matching.get("track_id"),
                         "class_name": matching["class_name"], "frame_id": self.frame_id}
        with self._lock:
            self.selected = selection
        return selection

    def register_selected(self, name: str, category: str | None = None) -> dict:
        if not name or not name.strip():
            raise ValueError("Enter a name for this asset")
        with self._lock:
            selection = dict(self.selected) if self.selected else None
            frame     = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is None or selection is None:
            raise ValueError("Select an object in the live camera first")
        box      = selection["bbox"]
        crop     = self._crop(frame, box)
        features = self.engine.extract_features(crop)
        tracker_mode   = selection["mode"]
        category_value = (category or selection.get("class_name") or "custom").strip().lower()
        profile        = {"version": 1, "samples": [features]}
        asset          = self.store.create_asset(name.strip(), category_value, tracker_mode, profile, box, None)

        # Save thumbnail (max 160 px, JPEG Q=75).
        target_dir = self.reference_dir / asset["asset_id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        _save_crop_thumbnail(target_dir / "reference_01.jpg", crop)

        # Seed identity cache.
        s = self._identity_cache.ensure(asset["asset_id"])
        s.last_box = box
        s.rolling_embeddings.append(features.get("embedding", []))
        s.confidence     = 1.0
        s.last_seen_mono = time.monotonic()
        s.gallery_size   = 1

        if tracker_mode == "detector" and selection.get("track_id") is not None:
            tid = int(selection["track_id"])
            self.track_assignments[tid]     = {"asset_id": asset["asset_id"], "similarity": None, "lost_since": None}
            self.track_feature_checked[tid] = time.monotonic()
        elif tracker_mode == "csrt":
            try:
                tracker   = self._new_csrt()
                bbox_xywh = (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))
                tracker.init(frame, bbox_xywh)
                with self._lock:
                    self.custom_trackers[asset["asset_id"]] = {
                        "tracker": tracker, "name": asset["name"],
                        "last_box": box, "failures": 0}
            except Exception as exc:
                self.store.set_status(asset["asset_id"], "UNVERIFIED", None)
                self.store.add_event(asset["asset_id"], "TRACKER_INIT_FAILED",
                                     f"Manual tracker could not start: {exc}")
        self.asset_last_seen_mono[asset["asset_id"]] = time.monotonic()
        self.asset_visible_boxes[asset["asset_id"]]  = box
        with self._lock:
            self.selected = None
        return self.store.get_asset(asset["asset_id"]) or asset

    def add_reference_from_selection(self, asset_id: str) -> dict:
        asset = self.store.get_asset(asset_id)
        if not asset:
            raise ValueError("Asset was not found")
        with self._lock:
            selection = dict(self.selected) if self.selected else None
            frame     = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is None or selection is None:
            raise ValueError("Select another view of the object first")
        crop     = self._crop(frame, selection["bbox"])
        features = self.engine.extract_features(crop)
        profile  = asset["profile"]

        current_scores = [self.engine.compare_features(features, s)
                          for s in profile.get("samples", [])]
        current_score  = max(current_scores, default=0.0)
        competing      = []
        for other in self.store.list_assets():
            if other["asset_id"] == asset_id:
                continue
            sc = [self.engine.compare_features(features, s)
                  for s in other.get("profile", {}).get("samples", [])]
            if sc:
                competing.append(max(sc))
        if current_score < 0.50 or (competing and max(competing) > current_score + 0.10):
            raise ValueError(
                "This view does not confidently match the selected asset profile. "
                "Select the item more tightly or try another angle; the saved profile was not changed.")

        samples = profile.setdefault("samples", [])
        samples.append(features)
        profile["samples"] = samples[-GALLERY_MAX:]
        self.store.update_profile(asset_id, profile)

        # Save thumbnail (max 3 manual reference thumbnails on disk).
        path  = self.reference_dir / asset_id
        path.mkdir(parents=True, exist_ok=True)
        count = len(profile["samples"])
        if count <= 3:
            _save_crop_thumbnail(path / f"reference_{count:02d}.jpg", crop)

        # Update identity cache.
        s = self._identity_cache.ensure(asset_id)
        s.rolling_embeddings.append(features.get("embedding", []))
        s.gallery_size = len(profile["samples"])

        if asset.get("tracker_mode") == "csrt":
            tracker  = self._new_csrt()
            bbox     = selection["bbox"]
            tracker.init(frame, (int(bbox[0]), int(bbox[1]),
                                  int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])))
            with self._lock:
                self.custom_trackers[asset_id] = {
                    "tracker": tracker, "name": asset["name"],
                    "last_box": bbox, "failures": 0}
            self.asset_last_seen_mono[asset_id] = time.monotonic()
            self.store.observe(asset_id, bbox, current_score)
            self.store.set_status(asset_id, "PRESENT", f"{asset['name']} visually reacquired")
        with self._lock:
            self.selected = None
        return self.store.get_asset(asset_id) or asset

    # ------------------------------------------------------------------
    # Archive / Hard Delete
    # ------------------------------------------------------------------

    def archive_asset(self, asset_id: str) -> bool:
        self._cleanup_live_state(asset_id)
        return self.store.deactivate_asset(asset_id)

    def hard_delete_asset(self, asset_id: str) -> bool:
        self._cleanup_live_state(asset_id)
        return self.store.hard_delete_asset(asset_id, reference_dir=self.reference_dir)

    def _cleanup_live_state(self, asset_id: str) -> None:
        """Remove all in-memory references to an asset."""
        self.track_assignments = {
            tid: e for tid, e in self.track_assignments.items()
            if e.get("asset_id") != asset_id}
        with self._lock:
            self.custom_trackers.pop(asset_id, None)
        self._identity_cache.remove(asset_id)
        self.asset_last_seen_mono.pop(asset_id, None)
        self.asset_visible_boxes.pop(asset_id, None)

    # ------------------------------------------------------------------
    # Monitoring mode
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        global _MODE
        if mode not in ("always", "day", "office", "night"):
            raise ValueError("Mode must be one of: always, day, office, night")
        _MODE = mode
        self.store.set_setting("mode", mode)

    def get_mode_info(self) -> dict:
        idle = _mode_is_idle()
        return {
            "mode":     _MODE,
            "idle":     idle,
            "day_start": _DAY_START,
            "day_end":   _DAY_END,
            "idle_fps":  _IDLE_FPS,
            "active_fps": self.target_infer_fps,
        }

    # ------------------------------------------------------------------
    # API state snapshot
    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            frame_age      = now - self.frame_timestamp if self.frame_timestamp else None
            camera_healthy = bool(self.camera_open and frame_age is not None and frame_age < 3.0)
            detections     = [dict(d) for d in self.detections]
            selected       = dict(self.selected) if self.selected else None
            dimensions     = {"width": self.frame_width, "height": self.frame_height}
            info = {
                "camera_index":            self.camera_index,
                "camera_healthy":          camera_healthy,
                "camera_error":            self.camera_error,
                "capture_fps":             round(self.capture_fps,     1),
                "infer_fps":               round(self.infer_fps,        1),
                "infer_latency_ms":        round(self.infer_latency_ms, 1),
                "frame_age_ms":            round(frame_age * 1000, 1) if frame_age is not None else None,
                "dimensions":              dimensions,
                "device":                  self.device_status,
                "gpu_available":           self.engine.gpu_available,
                "cuda_detected":           self.engine.cuda_detected,
                "compute_fallback_reason": self.engine.fallback_reason,
                "model_ready":             self.engine.ready,
                "model_error":             self.startup_error or self.engine.error,
                "detection_count":         len(detections),
                "target_infer_fps":        self.target_infer_fps,
                "missing_after_s":         self.missing_after_s,
                "selected":                selected,
                "monitoring":              self.get_mode_info(),
            }
        public_fields = ("asset_id", "name", "category", "tracker_mode", "status",
                         "last_seen", "last_similarity", "active",
                         "created_at", "updated_at", "anchor_box", "last_box")
        assets = []
        for a in self.store.list_assets():
            entry = {k: a.get(k) for k in public_fields}
            s     = self._identity_cache.get(a["asset_id"])
            entry["identity_confidence"]    = round(s.confidence, 3) if s else None
            entry["consecutive_matches"]    = s.consecutive_matches if s else 0
            entry["recovered_from_gallery"] = s.recovered_from_gallery if s else False
            entry["gallery_size"]           = s.gallery_size if s else len(a.get("profile", {}).get("samples", []))
            assets.append(entry)
        return {
            "runtime":    info,
            "detections": detections,
            "assets":     assets,
            "events":     self.store.list_events(30),
        }

    # ------------------------------------------------------------------
    # Annotated MJPEG
    # ------------------------------------------------------------------

    def annotated_frame(self) -> np.ndarray:
        with self._lock:
            frame        = self.latest_frame.copy() if self.latest_frame is not None else None
            detections   = [dict(d) for d in self.detections]
            selected     = dict(self.selected) if self.selected else None
            fps          = self.capture_fps
            infer_fps    = self.infer_fps
            latency      = self.infer_latency_ms
            device       = self.device_status
            camera_error = self.camera_error
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for USB webcam...",
                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 230), 2)
            cv2.putText(frame, camera_error or "Plug in the camera or select another camera index",
                        (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 190, 250), 2)
            return frame
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            matched   = bool(det.get("asset_id"))
            recovered = bool(det.get("recovered"))
            color     = (50, 205, 120) if matched else (242, 174, 63)
            if det.get("source") == "csrt":
                color = (210, 135, 255)
            if recovered:
                color = (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = det.get("asset_name") if matched else det.get("class_name", "object")
            if det.get("confidence") is not None and not matched:
                label += f" {det['confidence']:.2f}"
            elif matched and det.get("similarity") is not None:
                label += f" ·{det['similarity']:.2f}"
            if recovered:
                label += " ↩"
            label = str(label)[:58]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(0, y1 - th - 10)
            cv2.rectangle(frame, (x1, top),
                          (min(frame.shape[1] - 1, x1 + tw + 8), y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 24, 30), 2, cv2.LINE_AA)
        if selected:
            x1, y1, x2, y2 = [int(v) for v in selected["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), (12, 18, 29), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        idle_tag = " [IDLE]" if _mode_is_idle() else ""
        cv2.putText(frame,
                    f"CAP {fps:.1f} FPS   DET {infer_fps:.1f} FPS   {latency:.0f} ms   {device[:35]}{idle_tag}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 248, 252), 1, cv2.LINE_AA)
        return frame

    def mjpeg(self):
        while not self._stop.is_set():
            frame = self.annotated_frame()
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n"
                       + encoded.tobytes() + b"\r\n")
            time.sleep(1 / 24)
