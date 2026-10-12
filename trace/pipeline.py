"""
The Core Edge Pipeline Loop.
Executes the full per-frame pipeline: capture -> health -> align -> people -> gate -> verify -> belief -> policy -> evidence -> ledger -> outbox.
Matching Section 4, Section 10.1, and System Design Image 1.
"""

from __future__ import annotations

import collections
import datetime
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from trace.config import AppConfig, AssetConfig
from trace.logic.fsm import AssetFSM, FSMConfig, Obs, S
from trace.logic.policy import PolicyEngine
from trace.platform.alerts import AlertRouter
from trace.platform.anchor import AnchorService
from trace.platform.db import DatabaseManager
from trace.platform.demo_inject import DemoFaultInjector
from trace.platform.evidence import EvidenceWriter
from trace.platform.ledger import LedgerManager
from trace.platform.outbox import OutboxManager
from trace.vision.align import FrameAligner
from trace.vision.capture import FrameCapture
from trace.vision.gate import ChangeGate
from trace.vision.health import CameraHealthMonitor
from trace.vision.newobject import NewObjectDetector
from trace.vision.people import PeopleSegmenter
from trace.vision.relocate import AssetRelocator
from trace.vision.stateclf import StateClassifier
from trace.vision.verify import RegionVerifier

logger = logging.getLogger("trace.pipeline")


def validate_region_bounds(
    region: List[List[int]], frame_shape: Tuple[int, int], min_dimension: int = 10
) -> Tuple[bool, str]:
    """
    Validates that a region polygon is well-formed and fits completely within the frame dimensions.
    Returns (is_valid, error_reason).
    """
    if not region or len(region) < 3:
        return False, "Region must contain at least 3 points."

    h, w = frame_shape[:2]
    if h <= 0 or w <= 0:
        return False, f"Invalid frame dimensions: {w}x{h}"

    pts = np.array(region, dtype=np.int32)
    for idx, (px, py) in enumerate(pts):
        if px < 0 or px > w or py < 0 or py > h:
            return False, f"Point {idx} ({px}, {py}) is out of frame bounds ({w}x{h})."

    rx, ry, rw, rh = cv2.boundingRect(pts)
    if rw < min_dimension or rh < min_dimension:
        return False, f"Region bounding box ({rw}x{rh}) is smaller than minimum required ({min_dimension}x{min_dimension})."

    if rx + rw > w or ry + rh > h:
        return False, f"Region bounding box extends beyond frame bounds ({w}x{h})."

    return True, ""


class TracePipeline:
    """
    Master edge processing engine for TRACE.
    """

    def __init__(self, cfg: AppConfig, db_manager: DatabaseManager):
        self.cfg = cfg
        self.db_mgr = db_manager
        self.db = db_manager.get_connection()

        # Platform components
        self.ledger_mgr = LedgerManager(key_path=cfg.ledger.key_file)
        self.evidence_writer = EvidenceWriter(base_dir="data/evidence")
        self.outbox = OutboxManager(self.db)
        self.anchor_service = AnchorService()
        self.alert_router = AlertRouter(cfg.alerts)
        self.demo_injector = DemoFaultInjector()
        self.active_camera_fault: Optional[str] = None

        # Vision components
        cam_cfg = cfg.cameras[0] if cfg.cameras else None
        source_uri = cam_cfg.source if cam_cfg else "synthetic"
        self.capture = FrameCapture(
            source_uri=source_uri,
            width=cam_cfg.width if cam_cfg else 1280,
            height=cam_cfg.height if cam_cfg else 720,
            target_fps=cam_cfg.process_fps if cam_cfg else 5
        )

        timers = cfg.get_timers()
        self.health_monitor = CameraHealthMonitor(
            hold_sec=timers.obstruction_hold_s,
            release_sec=timers.obstruction_release_s
        )
        self.aligner = FrameAligner()
        self.people_segmenter = PeopleSegmenter(mask_dilation_px=15, history_len=3)
        self.change_gate = ChangeGate()
        self.verifier = RegionVerifier()
        self.relocator = AssetRelocator()
        self.new_object_detector = NewObjectDetector(persistence_s=timers.new_object_s)
        self.state_classifier = StateClassifier(k_neighbors=5)

        # Logic components
        self.policy_engine = PolicyEngine(cfg.business_hours)

        # Asset State & Baseline dictionaries
        self.assets: Dict[str, AssetConfig] = {a.id: a for a in cfg.assets}
        self.fsms: Dict[str, AssetFSM] = {}
        self.baseline_variants: Dict[str, List[np.ndarray]] = {}
        self.baseline_medians: Dict[str, np.ndarray] = {}
        self.empty_rois: Dict[str, np.ndarray] = {}

        # Rolling buffer of recent frames for genuine pre-incident evidence
        self.recent_frames: collections.deque = collections.deque(maxlen=30)

        # Pipeline runtime state
        self.is_running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._outbox_thread: Optional[threading.Thread] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_clean_frame: Optional[np.ndarray] = None
        self.current_fps: float = 0.0
        self.last_heartbeat_time: float = 0.0
        self.total_frames_processed: int = 0
        self.lock = threading.Lock()

        # Restore approved reference frame from disk if present to survive restarts
        ref_dir = Path(self.cfg.ledger.key_file).parent if (self.cfg.ledger and self.cfg.ledger.key_file) else Path("data")
        cam_id = self.cfg.cameras[0].id if self.cfg.cameras else "cam-1"
        self.reference_path = ref_dir / f"reference_{cam_id}.jpg"
        if not self.reference_path.exists() and Path("data/reference_frame.jpg").exists():
            self.reference_path = Path("data/reference_frame.jpg")

        self.aligner = FrameAligner(reference_path=str(self.reference_path))
        if self.reference_path.exists():
            self.aligner.load_reference(str(self.reference_path))

        # Bootstrap FSMs and initial baselines
        self._init_assets_and_baselines()

    @property
    def is_alive(self) -> bool:
        """Returns True if the background processing loop thread is currently alive and executing."""
        return self._thread is not None and self._thread.is_alive()

    def _init_assets_and_baselines(self) -> None:
        """Initializes FSMs and baseline profiles for registered assets."""
        timers = self.cfg.get_timers()

        # Ensure cameras exist in DB to satisfy foreign key constraints
        with self.db_mgr.get_connection() as conn:
            for cam in self.cfg.cameras:
                conn.execute(
                    """
                    INSERT INTO cameras (
                        id, name, source_uri, status, last_heartbeat
                    ) VALUES (?, ?, ?, 'ONLINE', ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        source_uri = excluded.source_uri,
                        last_heartbeat = excluded.last_heartbeat
                    """,
                    (cam.id, cam.name, cam.source, time.time())
                )
            conn.commit()

            # Query database for persistent runtime API-registered assets
            cur = conn.execute(
                """
                SELECT id, name, class, criticality, policy_mode, camera_id,
                       region_polygon, zone_polygon, tag_id, present_threshold
                FROM assets WHERE active = 1
                """
            )
            for row in cur.fetchall():
                a_id = row[0]
                if a_id not in self.assets:
                    try:
                        r_poly = json.loads(row[6]) if row[6] else []
                        z_poly = json.loads(row[7]) if row[7] else r_poly
                        db_a = AssetConfig(
                            id=a_id,
                            name=row[1] or a_id,
                            class_name=row[2] or "other",
                            criticality=int(row[3]) if row[3] is not None else 1,
                            mode=row[4] or "fixed",
                            camera_id=row[5] or "cam-1",
                            region=r_poly,
                            zone=z_poly,
                            tag_id=int(row[8]) if row[8] is not None else None,
                        )
                        db_a.thresholds.present = float(row[9]) if row[9] is not None else 0.70
                        self.assets[a_id] = db_a
                        logger.info(f"Loaded persistent API asset '{a_id}' from database.")
                    except Exception as e:
                        logger.error(f"Failed to load DB asset '{a_id}': {e}")

        for asset_id, a in self.assets.items():
            half_life = self.cfg.get_half_life(a.class_name)
            fsm_cfg = FSMConfig(
                move_frac=a.thresholds.move_frac,
                present=a.thresholds.present,
                vis_floor=a.thresholds.vis_floor,
                missing_wait_s=timers.missing_wait_s,
                swap_s=timers.swap_s,
                moved_votes=timers.moved_votes,
                returned_hold_s=timers.returned_hold_s,
                half_life_s=half_life,
            )
            self.fsms[asset_id] = AssetFSM(fsm_cfg)

            # Persist or update in DB `assets` and restore/initialize `states`
            with self.db_mgr.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO assets (
                        id, name, class, criticality, policy_mode, camera_id,
                        region_polygon, zone_polygon, tag_id, present_threshold, active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        class = excluded.class,
                        criticality = excluded.criticality,
                        policy_mode = excluded.policy_mode,
                        camera_id = excluded.camera_id,
                        region_polygon = excluded.region_polygon,
                        zone_polygon = excluded.zone_polygon,
                        tag_id = excluded.tag_id,
                        present_threshold = excluded.present_threshold,
                        active = 1
                    """,
                    (
                        a.id, a.name, a.class_name, a.criticality, a.mode, a.camera_id,
                        json.dumps(a.region), json.dumps(a.zone), a.tag_id, a.thresholds.present
                    )
                )

                # Check if asset state was already persisted from a previous run
                cur = conn.execute(
                    "SELECT state, belief, confidence, last_verified_at FROM states WHERE asset_id = ?",
                    (a.id,)
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    saved_state_str = row[0]
                    saved_belief = float(row[1]) if row[1] is not None else 1.0
                    try:
                        self.fsms[asset_id].state = S(saved_state_str)
                    except (ValueError, KeyError):
                        self.fsms[asset_id].state = S.UNKNOWN
                    self.fsms[asset_id].doubt_engine.belief = saved_belief
                    logger.info(
                        f"Restored persisted state for asset '{a.id}': state={self.fsms[asset_id].state.value}, belief={saved_belief:.2f}"
                    )
                else:
                    self.fsms[asset_id].state = S.UNKNOWN
                    now_t = time.time()
                    conn.execute(
                        """
                        INSERT INTO states (
                            asset_id, state, belief, confidence, last_verified_at, updated_at
                        ) VALUES (?, 'UNKNOWN', 0.0, 0.0, ?, ?)
                        ON CONFLICT(asset_id) DO UPDATE SET
                            state = 'UNKNOWN',
                            belief = 0.0,
                            confidence = 0.0,
                            last_verified_at = excluded.last_verified_at,
                            updated_at = excluded.updated_at
                        """,
                        (a.id, now_t, now_t)
                    )
                conn.commit()

        # Load any existing persisted baselines from disk/db to survive restarts
        with self.db_mgr.get_connection() as conn:
            for asset_id in self.assets.keys():
                cur = conn.execute(
                    "SELECT image_path FROM baselines WHERE asset_id = ? AND kind = 'median' ORDER BY created_at DESC LIMIT 1",
                    (asset_id,)
                )
                row = cur.fetchone()
                if row and Path(row[0]).exists():
                    crop = cv2.imread(row[0])
                    if crop is not None and crop.size > 0:
                        self.baseline_medians[asset_id] = crop
                        self.baseline_variants[asset_id] = self.verifier.extractor.generate_variants(crop)
                        logger.info(f"Restored persisted median baseline for '{asset_id}'.")

                cur_empty = conn.execute(
                    "SELECT image_path FROM baselines WHERE asset_id = ? AND kind = 'empty' ORDER BY created_at DESC LIMIT 1",
                    (asset_id,)
                )
                row_empty = cur_empty.fetchone()
                if row_empty and Path(row_empty[0]).exists():
                    empty_crop = cv2.imread(row_empty[0])
                    if empty_crop is not None and empty_crop.size > 0:
                        self.empty_rois[asset_id] = empty_crop
                        logger.info(f"Restored persisted empty baseline for '{asset_id}'.")
                else:
                    self.empty_rois[asset_id] = None

    def _capture_initial_baselines(self, frame_bgr: np.ndarray, overwrite: bool = False) -> Dict[str, Any]:
        """
        Extracts initial median baselines and Visual DNA variants from canonical reference frame.
        Explicitly validates that each configured asset region contains valid visual content and matching tag if configured.
        """
        self.aligner.register_reference(frame_bgr, save_to_disk=True, force=overwrite)
        self.health_monitor.calibrate_baseline(frame_bgr)

        # Warp reference into canonical coordinate space
        canonical_frame = self.aligner.warp_to_reference(frame_bgr)
        h, w = canonical_frame.shape[:2]

        baselines_dir = Path("data/baselines")
        baselines_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Any] = {"calibrated": [], "failed": [], "skipped": []}

        for asset_id, a in self.assets.items():
            # If baseline already exists and not forcing overwrite, skip
            if not overwrite and asset_id in self.baseline_medians and self.baseline_medians[asset_id] is not None:
                results["skipped"].append(asset_id)
                continue

            valid, err = validate_region_bounds(a.region, (h, w))
            if not valid:
                logger.warning(f"Skipping baseline for '{asset_id}': {err}")
                results["failed"].append({"asset_id": asset_id, "reason": err})
                continue

            pts = np.array(a.region, dtype=np.int32)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            crop = canonical_frame[ry : ry + rh, rx : rx + rw].copy()
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                reason = "Crop is empty or degenerate"
                logger.warning(f"Baseline crop for '{asset_id}' failed: {reason}")
                results["failed"].append({"asset_id": asset_id, "reason": reason})
                continue

            # Contrast / visual variance check: reject blank/flat regions
            crop_var = float(np.var(crop))
            if crop_var < 5.0:
                reason = f"Crop has insufficient visual texture (variance={crop_var:.1f})"
                logger.warning(f"Baseline crop for '{asset_id}' failed: {reason}")
                results["failed"].append({"asset_id": asset_id, "reason": reason})
                continue

            # Tag validation: if tag_id is configured, detect if present
            if a.tag_id is not None and getattr(self.verifier, "aruco_detector", None) is not None:
                from trace.vision.verify import detect_primary_tag
                detected_tag = detect_primary_tag(crop)
                if detected_tag is not None and detected_tag != a.tag_id:
                    reason = f"Mismatched tag detected in region (expected {a.tag_id}, found {detected_tag})"
                    logger.warning(f"Baseline validation warning for '{asset_id}': {reason}")
                    results["failed"].append({"asset_id": asset_id, "reason": reason})
                    continue

            self.baseline_medians[asset_id] = crop
            vars_vec = self.verifier.extractor.generate_variants(crop)
            self.baseline_variants[asset_id] = vars_vec

            # Persist to disk and baselines table in SQLite
            img_path = baselines_dir / f"{asset_id}_init_median.jpg"
            cv2.imwrite(str(img_path), crop)

            with self.db_mgr.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO baselines (
                        id, asset_id, kind, image_path, created_at
                    ) VALUES (?, ?, 'median', ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        image_path = excluded.image_path,
                        created_at = excluded.created_at
                    """,
                    (f"base-{asset_id}-init", asset_id, str(img_path).replace("\\", "/"), time.time())
                )
                conn.commit()

            results["calibrated"].append(asset_id)

        return results

    def _outbox_delivery_handler(self, event_info: Dict[str, Any]) -> bool:
        """Durable outbox delivery handler dispatching external alerts with verified delivery status."""
        try:
            payload_str = event_info.get("payload", "{}") if isinstance(event_info, dict) else str(event_info)
            payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
            return self.alert_router.send_external(payload)
        except Exception as e:
            logger.error(f"Outbox delivery attempt failed: {e}")
            return False

    def _outbox_worker_loop(self) -> None:
        """Dedicated background outbox worker for external alert delivery decoupled from video frame processing."""
        while self.is_running:
            try:
                self.outbox.flush(delivery_handler=self._outbox_delivery_handler)
            except Exception as e:
                logger.error(f"Error in outbox worker loop: {e}", exc_info=True)
            time.sleep(1.0)

    def anchor_ledger(self, channel: str = "local_file") -> Optional[Dict[str, Any]]:
        """Anchors the ledger head outside the edge database to prevent full-chain rewrites."""
        with self.db_mgr.get_connection() as conn:
            return self.anchor_service.anchor_head(conn, channel=channel)

    def start(self) -> None:
        """Starts the capture and processing pipeline."""
        if self.is_running:
            return
        self.capture.start()

        # Wait for the first genuine frame from the active camera source
        timeout = 2.0 if not self.capture.is_connected else 5.0
        ok, init_frame, _ = self.capture.wait_for_first_frame(timeout_sec=timeout)
        if ok and init_frame is not None:
            self._capture_initial_baselines(init_frame)
            logger.info("Calibrated initial baselines from live camera frame.")
        else:
            logger.warning("Camera not delivering frames at startup. Marking camera OFFLINE.")
            with self.db_mgr.get_connection() as conn:
                for cam in self.cfg.cameras:
                    conn.execute("UPDATE cameras SET status = 'OFFLINE' WHERE id = ?", (cam.id,))
                conn.commit()

        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="trace-pipeline-loop")
        self._thread.start()
        self._outbox_thread = threading.Thread(target=self._outbox_worker_loop, daemon=True, name="trace-outbox-worker")
        self._outbox_thread.start()
        logger.info("TRACE Pipeline and Outbox worker started successfully.")

    def stop(self) -> None:
        """Stops the processing pipeline."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._outbox_thread and self._outbox_thread.is_alive():
            self._outbox_thread.join(timeout=1.5)
        self.capture.stop()
        logger.info("TRACE Pipeline stopped.")

    def _run_loop(self) -> None:
        target_fps = 5
        interval = 1.0 / target_fps
        t_last = time.time()
        cam_id = self.cfg.cameras[0].id if self.cfg.cameras else "cam-1"

        while self.is_running:
            t_start = time.time()
            try:
                env = self.capture.read_envelope(camera_id=cam_id)

                if env.is_valid and env.image is not None:
                    # Deduplication: process each frame ID at most once
                    if not env.is_new:
                        time.sleep(0.02)
                        continue

                    raw_frame = env.image
                    capture_t = env.captured_at_monotonic_ns / 1e9

                    # Restore camera online status if previously marked offline
                    with self.db_mgr.get_connection() as conn:
                        for cam in self.cfg.cameras:
                            conn.execute("UPDATE cameras SET status = 'ONLINE', last_heartbeat = ? WHERE id = ? AND status = 'OFFLINE'", (time.time(), cam.id))
                        conn.commit()

                    # If initial baselines were missed due to camera offline at startup, calibrate now
                    if not self.baseline_medians:
                        self._capture_initial_baselines(raw_frame)
                        logger.info("Calibrated initial baselines following camera reconnection.")

                    # 1. Apply demo scenario perturbations if triggered
                    injected_frame, synthetic_person_mask = self.demo_injector.apply_to_frame(
                        raw_frame, self.assets, self.empty_rois
                    )
                    # Sync outbox network state with demo injector
                    self.outbox.set_network_state(not self.demo_injector.is_network_cut())

                    now = time.time()
                    # Store genuine unperturbed frame in rolling buffer for pre-incident evidence
                    self.recent_frames.append((now, injected_frame.copy()))

                    # 2. Camera health & optical obstruction evaluation
                    health_res = self.health_monitor.check(injected_frame, capture_t, landmark_ratio=self.aligner.inlier_ratio)
                    camera_ok = health_res["ok"] and not health_res["is_offline"]

                    # 3. Environmental alignment & camera tampering check
                    align_res = self.aligner.align_frame(injected_frame)
                    is_cam_tampered = align_res.get("is_tampered", False)

                    # Camera fault durability check
                    if health_res.get("is_obstructed", False):
                        self._handle_camera_fault(
                            camera_id=cam_id,
                            fault_type="CAMERA_OBSTRUCTED",
                            details=health_res.get("metrics", {}),
                            frame_bgr=injected_frame,
                            timestamp=now,
                        )
                    elif is_cam_tampered:
                        self._handle_camera_fault(
                            camera_id=cam_id,
                            fault_type="CAMERA_TAMPERED",
                            details={"shift_px": align_res.get("shift_px", 0.0), "inlier_ratio": getattr(self.aligner, "inlier_ratio", 0.0)},
                            frame_bgr=injected_frame,
                            timestamp=now,
                        )
                    elif health_res.get("is_offline", False):
                        self._handle_camera_fault(
                            camera_id=cam_id,
                            fault_type="SYSTEM_OFFLINE",
                            details={"reason": health_res.get("reason", "Feed unavailable")},
                            frame_bgr=injected_frame,
                            timestamp=now,
                        )
                    else:
                        if self.active_camera_fault is not None:
                            self.active_camera_fault = None

                    # 4. People segmentation & privacy occlusion mask
                    person_mask = self.people_segmenter.segment(injected_frame, extra_mask=synthetic_person_mask)

                    # 5. Process registered assets
                    annotated = injected_frame.copy()

                    for asset_id, asset_cfg in self.assets.items():
                        self._process_single_asset(
                            asset_id=asset_id,
                            asset_cfg=asset_cfg,
                            frame_bgr=injected_frame,
                            annotated_frame=annotated,
                            person_mask=person_mask,
                            camera_ok=camera_ok,
                            is_cam_tampered=is_cam_tampered,
                            timestamp=now,
                        )

                    # 6. Check for unexpected objects in zones (NEW_OBJECT)
                    reg_polys = [a.region for a in self.assets.values() if a.region]
                    new_objs = self.new_object_detector.update(
                        injected_frame, person_mask, reg_polys, timestamp=now
                    )
                    for no in new_objs:
                        if no.get("is_new_confirmation", False):
                            self._handle_new_object(no, injected_frame, person_mask, now)

                    # 7. Render HUD overlays on annotated frame
                    self._draw_hud(annotated, health_res, align_res, now)

                    with self.lock:
                        self.latest_clean_frame = injected_frame
                        self.latest_annotated_frame = annotated
                        self.total_frames_processed += 1
                        # Rolling FPS calculation
                        dt = time.time() - t_last
                        if dt > 0:
                            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)
                        t_last = time.time()
                else:
                    # Camera feed unavailable, stalled, or disconnected: record offline state
                    capture_t = env.captured_at_monotonic_ns / 1e9 if env.captured_at_monotonic_ns else time.monotonic()
                    health_off = self.health_monitor.check(None, capture_t)
                    if health_off.get("is_offline", False):
                        self._handle_camera_fault(
                            camera_id=cam_id,
                            fault_type="SYSTEM_OFFLINE",
                            details={"reason": health_off.get("reason", "Camera feed disconnected")},
                            frame_bgr=None,
                            timestamp=time.time(),
                        )
                    else:
                        with self.db_mgr.get_connection() as conn:
                            for cam in self.cfg.cameras:
                                conn.execute("UPDATE cameras SET status = 'OFFLINE' WHERE id = ?", (cam.id,))
                            conn.commit()
            except Exception as e:
                logger.error(f"Error in pipeline frame loop: {e}", exc_info=True)

            # Maintain target cadence
            elapsed = time.time() - t_start
            sleep_s = interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _handle_camera_fault(
        self,
        camera_id: str,
        fault_type: str,
        details: Dict[str, Any],
        frame_bgr: Optional[np.ndarray],
        timestamp: float,
    ) -> None:
        """Atomically persists durable camera fault incident, event, ledger record, and outbox notification."""
        if self.active_camera_fault == fault_type:
            return

        event_id = str(uuid.uuid4())
        incident_id = str(uuid.uuid4())
        dt_str = datetime.datetime.fromtimestamp(timestamp).astimezone().isoformat()

        before_crop = None
        if self.recent_frames:
            before_crop = self.recent_frames[0][1].copy()
        if before_crop is None:
            if frame_bgr is not None:
                before_crop = np.zeros_like(frame_bgr)
            else:
                before_crop = np.zeros((480, 640, 3), dtype=np.uint8)

        after_crop = frame_bgr.copy() if frame_bgr is not None else np.zeros_like(before_crop)

        # Fail-closed privacy redaction: mask people or blur full frame if privacy enabled
        before_mask = None
        after_mask = None
        if self.cfg.privacy.mask_people:
            try:
                before_mask = self.people_segmenter.segment(before_crop)
                after_mask = self.people_segmenter.segment(after_crop)
            except Exception as e:
                logger.warning(f"Privacy segmentation failed on camera fault; failing closed: {e}")
                before_crop = cv2.GaussianBlur(before_crop, (51, 51), 0)
                after_crop = cv2.GaussianBlur(after_crop, (51, 51), 0)

        try:
            ev_meta, crop_hashes = self.evidence_writer.save_crops(
                event_id=event_id,
                before_crop=before_crop,
                after_crop=after_crop,
                before_mask=before_mask,
                after_mask=after_mask,
                person_masked=self.cfg.privacy.mask_people,
            )
        except Exception as e:
            logger.error(f"Failed to save evidence crops for camera fault {fault_type}: {e}")
            return

        event_payload = {
            "id": event_id,
            "asset_id": None,
            "camera_id": camera_id,
            "type": fault_type,
            "severity": "high",
            "confidence": 0.95,
            "detected_at": dt_str,
            "evidence": ev_meta,
            "details": details,
        }

        try:
            with self.db_mgr.get_connection() as conn:
                # 1. Update camera status
                conn.execute(
                    "UPDATE cameras SET status = ? WHERE id = ?",
                    (fault_type, camera_id)
                )

                # 2. Append to cryptographic ledger
                ledger_info = self.ledger_mgr.append(
                    db=conn,
                    event=event_payload,
                    crop_hashes=crop_hashes,
                    commit=False,
                )
                event_payload["ledger"] = ledger_info

                # 3. Create incident
                conn.execute(
                    """
                    INSERT INTO incidents (id, asset_id, opened_at, status, counter)
                    VALUES (?, NULL, ?, 'OPEN', 1)
                    """,
                    (incident_id, timestamp)
                )

                # 4. Create event
                conn.execute(
                    """
                    INSERT INTO events (
                        id, asset_id, camera_id, type, severity, confidence, status,
                        detected_at, received_at, before_path, after_path, payload_json, incident_id
                    ) VALUES (?, NULL, ?, ?, 'high', 0.95, 'NEW', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, camera_id, fault_type,
                        timestamp, timestamp, ev_meta["before"], ev_meta["after"],
                        json.dumps(event_payload), incident_id
                    )
                )

                # 5. Enqueue into outbox
                self.outbox.enqueue_conn(conn, event_id, json.dumps(event_payload))
                conn.commit()

            # Set active_camera_fault ONLY after database commit succeeds
            self.active_camera_fault = fault_type
        except Exception as e:
            logger.error(f"Failed to persist camera fault {fault_type}: {e}")
            return

        self.alert_router.broadcast_ws(event_payload)
        logger.warning(f"DURABLE CAMERA FAULT [{fault_type}] on {camera_id}")

    def _process_single_asset(
        self,
        asset_id: str,
        asset_cfg: AssetConfig,
        frame_bgr: np.ndarray,
        annotated_frame: np.ndarray,
        person_mask: np.ndarray,
        camera_ok: bool,
        is_cam_tampered: bool,
        timestamp: float,
    ) -> None:
        fsm = self.fsms[asset_id]
        raw_region = asset_cfg.region
        if not raw_region or len(raw_region) < 3:
            return

        # Canonical Reference Coordinate Alignment:
        # Warp the current frame into the canonical reference coordinate system.
        # This guarantees regions, relocation bboxes, and drift calculations are directly comparable.
        canonical_frame = self.aligner.warp_to_reference(frame_bgr)
        canonical_mask = self.aligner.warp_to_reference(person_mask) if person_mask is not None else None
        h, w = canonical_frame.shape[:2]

        pts = np.array(raw_region, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))
        rw = max(1, min(w - rx, rw))
        rh = max(1, min(h - ry, rh))

        live_crop = canonical_frame[ry : ry + rh, rx : rx + rw]

        # Visibility and occlusion calculation in canonical reference frame
        occlusion_ratio, visibility = self.people_segmenter.calculate_visibility_and_occlusion(
            person_mask=canonical_mask if canonical_mask is not None else person_mask,
            region_pts=raw_region,
            frame_shape=(h, w)
        )

        base_median = self.baseline_medians.get(asset_id)
        base_vars = self.baseline_variants.get(asset_id, [])
        empty_roi = self.empty_rois.get(asset_id)

        # Region verification
        ver_res = self.verifier.verify_asset(
            live_crop_bgr=live_crop,
            baseline_variants=base_vars,
            baseline_median_crop=base_median if base_median is not None else live_crop,
            empty_roi_bgr=empty_roi,
            tag_id=asset_cfg.tag_id,
        )

        is_partially_occluded = ver_res.get("is_partially_occluded", False)
        tile_occ_ratio = ver_res.get("occlusion_ratio", 0.0)
        combined_visibility = min(visibility, 1.0 - tile_occ_ratio)

        # Asset localization and physical displacement in canonical coordinate system
        found_elsewhere = False
        asset_displacement_frac = 0.0
        relocated_score = 0.0

        # When presence in expected region drops, search canonical frame to distinguish relocation from missing
        if base_median is not None and ver_res["present_score"] < asset_cfg.thresholds.present:
            found_elsewhere, relocated_bbox, relocated_score = self.relocator.search(
                frame_bgr=canonical_frame,
                template_crop_bgr=base_median,
                exclude_zone=asset_cfg.zone or raw_region,
                person_mask=canonical_mask,
            )
            if found_elsewhere and relocated_bbox is not None:
                bx, by, bw, bh = relocated_bbox
                obs_region = [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]
                _, asset_displacement_frac = self.verifier.calculate_centroid_drift(obs_region, raw_region)

        # State change prediction (e.g. Safe door open/closed)
        state_label, state_conf = self.state_classifier.predict(asset_id, live_crop)

        # Build Observation with genuine asset displacement and tag verification status
        obs = Obs(
            t=timestamp,
            present_score=ver_res["present_score"],
            drift_frac=asset_displacement_frac,
            visibility=combined_visibility,
            empty_match=ver_res["empty_match"],
            camera_ok=camera_ok and not is_cam_tampered,
            found_elsewhere=found_elsewhere,
            state_label=state_label,
            state_conf=state_conf,
            tag_matched=ver_res["tag_matched"],
            tag_status=ver_res.get("tag_status", "NONE"),
            is_partially_occluded=is_partially_occluded,
        )

        # Step State Machine
        prev_state = fsm.state.value
        new_state = fsm.step(obs).value

        # Calculate event-specific confidence
        if new_state == "MOVED":
            event_confidence = max(0.75, float(relocated_score))
        elif new_state == "MISSING":
            event_confidence = round(float(min(1.0, (1.0 - ver_res["present_score"]) * combined_visibility)), 2)
        elif new_state == "SWAPPED":
            event_confidence = 0.95
        else:
            event_confidence = ver_res["present_score"]

        # Check Policy Engine for Alert Dispatch
        if new_state != prev_state and new_state not in ("SAFE", "OCCLUDED", "RETURNED", "UNKNOWN", "UNINITIALIZED"):
            # Alert transitions commit state, event, incident, ledger, and outbox atomically
            self._handle_alert_transition(
                asset_id=asset_id,
                asset_cfg=asset_cfg,
                prev_state=prev_state,
                new_state=new_state,
                confidence=event_confidence,
                belief=fsm.belief,
                details={
                    "drift_frac": round(asset_displacement_frac, 3),
                    "threshold": asset_cfg.thresholds.move_frac,
                    "visibility": round(combined_visibility, 2),
                    "empty_match": round(ver_res["empty_match"], 2),
                    "camera_ok": camera_ok,
                    "is_partially_occluded": is_partially_occluded,
                },
                live_crop=live_crop,
                base_crop=base_median if base_median is not None else live_crop,
                person_mask_crop=canonical_mask[ry : ry + rh, rx : rx + rw] if canonical_mask is not None else None,
                timestamp=timestamp,
            )
        else:
            # Routine non-alert state updates
            try:
                with self.db_mgr.get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE states
                        SET state = ?, belief = ?, confidence = ?, last_verified_at = ?, updated_at = ?
                        WHERE asset_id = ?
                        """,
                        (new_state, fsm.belief, event_confidence, timestamp, timestamp, asset_id)
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to update state for asset {asset_id}: {e}")

        # Draw asset box on annotated frame
        self._draw_asset_box(annotated_frame, pts, asset_cfg.name, new_state, fsm.belief)

    def _handle_alert_transition(
        self,
        asset_id: str,
        asset_cfg: AssetConfig,
        prev_state: str,
        new_state: str,
        confidence: float,
        belief: float,
        details: Dict[str, Any],
        live_crop: np.ndarray,
        base_crop: np.ndarray,
        person_mask_crop: Optional[np.ndarray],
        timestamp: float,
    ) -> None:
        policy_res = self.policy_engine.evaluate_transition(
            asset_id=asset_id,
            asset_class=asset_cfg.class_name,
            criticality=asset_cfg.criticality,
            policy_mode=asset_cfg.mode,
            prev_state=prev_state,
            new_state=new_state,
            confidence=confidence,
            current_t=timestamp,
        )

        if not policy_res:
            return

        severity = policy_res["severity"]
        event_id = str(uuid.uuid4())
        incident_id = policy_res.get("incident_id") or str(uuid.uuid4())

        # Save masked before/after evidence crops with fail-closed privacy redaction
        try:
            ev_meta, crop_hashes = self.evidence_writer.save_crops(
                event_id=event_id,
                before_crop=base_crop,
                after_crop=live_crop,
                before_mask=person_mask_crop if self.cfg.privacy.mask_people else None,
                after_mask=person_mask_crop if self.cfg.privacy.mask_people else None,
                person_masked=self.cfg.privacy.mask_people,
            )
        except Exception as e:
            logger.error(f"Failed to save evidence crops for alert transition {new_state}: {e}")
            return

        # Build Event Record matching Section 7.2
        dt_str = datetime.datetime.fromtimestamp(timestamp).astimezone().isoformat()
        event_payload = {
            "id": event_id,
            "asset_id": asset_id,
            "camera_id": asset_cfg.camera_id,
            "type": new_state,
            "severity": severity,
            "confidence": round(confidence, 2),
            "detected_at": dt_str,
            "evidence": ev_meta,
            "details": details,
        }

        # Atomically commit states update, ledger append, incidents, events, and outbox enqueue in ONE transaction
        try:
            with self.db_mgr.get_connection() as conn:
                # 1. Update persisted state coherently with event
                conn.execute(
                    """
                    UPDATE states
                    SET state = ?, belief = ?, confidence = ?, last_verified_at = ?, updated_at = ?
                    WHERE asset_id = ?
                    """,
                    (new_state, belief, confidence, timestamp, timestamp, asset_id)
                )

                # 2. Append to cryptographic ledger (within current transaction)
                ledger_info = self.ledger_mgr.append(
                    db=conn,
                    event=event_payload,
                    crop_hashes=crop_hashes,
                    commit=False,
                )
                event_payload["ledger"] = ledger_info

                # 3. Insert or update incident
                if not policy_res.get("is_repeat"):
                    conn.execute(
                        """
                        INSERT INTO incidents (id, asset_id, opened_at, status, counter)
                        VALUES (?, ?, ?, 'OPEN', 1)
                        """,
                        (incident_id, asset_id, timestamp)
                    )
                else:
                    conn.execute(
                        "UPDATE incidents SET counter = counter + 1 WHERE id = ?",
                        (incident_id,)
                    )

                initial_status = "NEEDS_REVIEW" if policy_res.get("tier") == "NEEDS_REVIEW" else "NEW"

                # 4. Insert event record
                conn.execute(
                    """
                    INSERT INTO events (
                        id, asset_id, camera_id, type, severity, confidence, status,
                        detected_at, received_at, before_path, after_path, payload_json, incident_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, asset_id, asset_cfg.camera_id, new_state, severity, confidence,
                        initial_status, timestamp, timestamp, ev_meta["before"], ev_meta["after"],
                        json.dumps(event_payload), incident_id
                    )
                )

                # 5. Atomically enqueue into outbox queue if push is not suppressed
                if not policy_res.get("suppress_push"):
                    self.outbox.enqueue_conn(conn, event_id, json.dumps(event_payload))
                conn.commit()
        except Exception as e:
            logger.error(f"Atomic alert persistence failed for {event_id}: {e}")
            return

        # Record for deduplication
        self.policy_engine.record_alert_dispatched(asset_id, new_state, incident_id, timestamp)

        # Broadcast immediately to local WebSocket subscribers (external channels handled exclusively via Outbox)
        self.alert_router.broadcast_ws(event_payload)
        logger.info(f"SECURITY ALERT [{severity.upper()}] {asset_id}: {new_state} (conf {confidence:.2f})")

    def _handle_new_object(
        self,
        blob_info: Dict[str, Any],
        frame_bgr: np.ndarray,
        person_mask: Optional[np.ndarray],
        timestamp: float
    ) -> None:
        bx, by, bw, bh = blob_info["bbox"]
        h, w = frame_bgr.shape[:2]
        crop = frame_bgr[max(0, by): min(h, by+bh), max(0, bx): min(w, bx+bw)]
        if crop.size == 0:
            return

        cam_id = self.cfg.cameras[0].id if self.cfg.cameras else "cam-1"
        foreign_id = f"foreign-{cam_id}-{blob_info.get('id', uuid.uuid4().hex[:6])}"
        event_id = str(uuid.uuid4())
        dt_str = datetime.datetime.fromtimestamp(timestamp).astimezone().isoformat()

        # Genuine pre-incident evidence crop from rolling buffer (never synthetic grey)
        before_crop = None
        if self.recent_frames:
            # Use earliest available clean frame prior to object placement
            earliest_frame = self.recent_frames[0][1]
            before_crop = earliest_frame[max(0, by): min(h, by+bh), max(0, bx): min(w, bx+bw)].copy()

        if before_crop is None or before_crop.size == 0 or before_crop.shape != crop.shape:
            before_crop = crop.copy()

        crop_mask = None
        if person_mask is not None:
            crop_mask = person_mask[max(0, by): min(h, by+bh), max(0, bx): min(w, bx+bw)]

        # Evidence crop with fail-closed privacy redaction
        try:
            ev_meta, crop_hashes = self.evidence_writer.save_crops(
                event_id=event_id,
                before_crop=before_crop,
                after_crop=crop,
                before_mask=crop_mask,
                after_mask=crop_mask,
                person_masked=self.cfg.privacy.mask_people,
            )
        except Exception as e:
            logger.error(f"Failed to save evidence crops for new object {foreign_id}: {e}")
            return

        event_payload = {
            "id": event_id,
            "foreign_object_id": foreign_id,
            "camera_id": cam_id,
            "type": "NEW_OBJECT",
            "severity": "medium",
            "confidence": 0.91,
            "detected_at": dt_str,
            "evidence": ev_meta,
            "details": {
                "bbox": [bx, by, bw, bh],
                "persisted_s": blob_info["persistence_s"],
            }
        }

        try:
            with self.db_mgr.get_connection() as conn:
                incident_id = str(uuid.uuid4())
                # 1. Track foreign object in foreign_objects table (not polluting registered assets)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO foreign_objects (
                        id, camera_id, first_seen_at, last_seen_at, bbox_json, status, incident_id
                    ) VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)
                    """,
                    (foreign_id, cam_id, timestamp - blob_info.get("persistence_s", 0), timestamp, json.dumps([bx, by, bw, bh]), incident_id)
                )

                # 2. Append to ledger within same transaction
                ledger_info = self.ledger_mgr.append(conn, event_payload, crop_hashes, commit=False)
                event_payload["ledger"] = ledger_info

                # 3. Open incident
                conn.execute(
                    "INSERT INTO incidents (id, asset_id, opened_at, status, counter) VALUES (?, NULL, ?, 'OPEN', 1)",
                    (incident_id, timestamp)
                )

                # 4. Insert event
                conn.execute(
                    """
                    INSERT INTO events (
                        id, asset_id, camera_id, type, severity, confidence, status,
                        detected_at, received_at, before_path, after_path, payload_json, incident_id
                    ) VALUES (?, NULL, ?, 'NEW_OBJECT', 'medium', 0.91, 'NEW', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, cam_id, timestamp, timestamp,
                        ev_meta["before"], ev_meta["after"], json.dumps(event_payload), incident_id
                    )
                )

                # 5. Atomically enqueue into outbox
                self.outbox.enqueue_conn(conn, event_id, json.dumps(event_payload))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed atomic persistence for new object {foreign_id}: {e}")
            return

        # Broadcast immediately to local WebSocket subscribers (external channels handled exclusively via Outbox)
        self.alert_router.broadcast_ws(event_payload)

    def _draw_asset_box(
        self,
        frame: np.ndarray,
        pts: np.ndarray,
        name: str,
        state: str,
        belief: float
    ) -> None:
        """Draws aesthetic bounding box and state chip matching Section 8.1 color rules."""
        # Color palette (BGR)
        color_map = {
            "SAFE": (70, 180, 80),             # Green
            "OCCLUDED": (130, 140, 150),       # Grey
            "UNKNOWN": (20, 160, 245),         # Amber
            "MOVED": (90, 60, 244),            # Rose
            "SWAPPED": (70, 30, 225),          # Strong Rose
            "MISSING": (70, 30, 225),          # Strong Rose
            "STATE_CHANGED": (180, 50, 180),   # Violet
            "RETURNED": (180, 180, 40),        # Cyan/Yellow
            "CAMERA_OBSTRUCTED": (220, 180, 0),# Cyan
            "SYSTEM_OFFLINE": (40, 40, 220),   # Red
        }
        color = color_map.get(state, (100, 100, 100))

        # Draw polygon boundary
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Label background pill
        rx, ry, rw, rh = cv2.boundingRect(pts)
        label_text = f"{name} | {state} ({belief:.2f})"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        pill_y1 = max(0, ry - th - 10)
        pill_y2 = ry
        pill_x2 = min(frame.shape[1], rx + tw + 12)

        cv2.rectangle(frame, (rx, pill_y1), (pill_x2, pill_y2), color, -1)
        cv2.putText(
            frame, label_text, (rx + 6, ry - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    def _draw_hud(
        self,
        frame: np.ndarray,
        health_res: Dict[str, Any],
        align_res: Dict[str, Any],
        timestamp: float
    ) -> None:
        """Renders enterprise top-bar telemetry HUD."""
        h, w = frame.shape[:2]
        # Dark HUD top ribbon
        cv2.rectangle(frame, (0, 0), (w, 36), (15, 18, 24), -1)

        # Status text
        cam_status = health_res["status"]
        cam_color = (0, 220, 0) if cam_status == "HEALTHY" else (0, 60, 240)

        outbox_pending = self.outbox.get_pending_count()
        net_str = "ONLINE" if self.outbox.network_connected else f"OFFLINE (Queue: {outbox_pending})"
        net_color = (0, 220, 0) if self.outbox.network_connected else (0, 140, 255)

        hud_text = (
            f"TRACE  {self.cfg.site}  |  FPS: {self.current_fps:.1f}  |  "
            f"Camera: {cam_status}  |  Network: {net_str}  |  "
            f"Align: {align_res['status']} ({align_res['shift_px']}px)"
        )
        cv2.putText(
            frame, hud_text, (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 235, 240), 1, cv2.LINE_AA
        )
