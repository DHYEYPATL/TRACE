"""
The Core Edge Pipeline Loop.
Executes the full per-frame pipeline: capture -> health -> align -> people -> gate -> verify -> belief -> policy -> evidence -> ledger -> outbox.
Matching Section 4, Section 10.1, and System Design Image 1.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from trace.config import AppConfig, AssetConfig
from trace.logic.fsm import AssetFSM, FSMConfig, Obs, S
from trace.logic.policy import PolicyEngine
from trace.platform.alerts import AlertRouter
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
        self.alert_router = AlertRouter(cfg.alerts)
        self.demo_injector = DemoFaultInjector()

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

        # Pipeline runtime state
        self.is_running: bool = False
        self._thread: Optional[threading.Thread] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_clean_frame: Optional[np.ndarray] = None
        self.current_fps: float = 0.0
        self.last_heartbeat_time: float = 0.0
        self.total_frames_processed: int = 0
        self.lock = threading.Lock()

        # Bootstrap FSMs and initial synthetic baselines
        self._init_assets_and_baselines()

    def _init_assets_and_baselines(self) -> None:
        """Initializes FSMs and baseline profiles for registered assets."""
        timers = self.cfg.get_timers()

        # Ensure cameras exist in DB to satisfy foreign key constraints
        with self.db_mgr.get_connection() as conn:
            for cam in self.cfg.cameras:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cameras (
                        id, name, source_uri, status, last_heartbeat
                    ) VALUES (?, ?, ?, 'ONLINE', ?)
                    """,
                    (cam.id, cam.name, cam.source, time.time())
                )
            conn.commit()

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

            # Persist in DB `assets` and `states`
            with self.db_mgr.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO assets (
                        id, name, class, criticality, policy_mode, camera_id,
                        region_polygon, zone_polygon, tag_id, present_threshold, active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        a.id, a.name, a.class_name, a.criticality, a.mode, a.camera_id,
                        json.dumps(a.region), json.dumps(a.zone), a.tag_id, a.thresholds.present
                    )
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO states (
                        asset_id, state, belief, confidence, last_verified_at, updated_at
                    ) VALUES (?, 'SAFE', 1.0, 1.0, ?, ?)
                    """,
                    (a.id, time.time(), time.time())
                )
                conn.commit()

        # Seed few-shot state classifier with mock safe door open/closed samples
        self.state_classifier.seed_default_branch_samples("safe-02")

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
                        self.empty_rois[asset_id] = np.full_like(crop, 160)

    def _capture_initial_baselines(self, frame_bgr: np.ndarray) -> None:
        """Extracts initial median baselines and Visual DNA variants from reference frame."""
        self.aligner.register_reference(frame_bgr)
        self.health_monitor.calibrate_baseline(frame_bgr)

        baselines_dir = Path("data/baselines")
        baselines_dir.mkdir(parents=True, exist_ok=True)

        for asset_id, a in self.assets.items():
            if not a.region or len(a.region) < 3:
                continue

            pts = np.array(a.region, dtype=np.int32)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            crop = frame_bgr[ry : ry + rh, rx : rx + rw].copy()
            self.baseline_medians[asset_id] = crop

            # Visual DNA variants (brightness, contrast, blur)
            vars_vec = self.verifier.extractor.generate_variants(crop)
            self.baseline_variants[asset_id] = vars_vec

            # Generate synthetic empty baseline (clean wall/desk background behind asset)
            empty_crop = np.full_like(crop, 160)
            self.empty_rois[asset_id] = empty_crop

            # Persist to disk and baselines table in SQLite
            img_path = baselines_dir / f"{asset_id}_init_median.jpg"
            cv2.imwrite(str(img_path), crop)

            with self.db_mgr.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO baselines (
                        id, asset_id, kind, image_path, created_at
                    ) VALUES (?, ?, 'median', ?, ?)
                    """,
                    (f"base-{asset_id}-init", asset_id, str(img_path).replace("\\", "/"), time.time())
                )
                conn.commit()

    def start(self) -> None:
        """Starts the capture and processing pipeline."""
        if self.is_running:
            return
        self.capture.start()

        # Wait for the first genuine frame from the active camera source
        ok, init_frame, _ = self.capture.wait_for_first_frame(timeout_sec=5.0)
        if ok and init_frame is not None:
            self._capture_initial_baselines(init_frame)
            logger.info("Calibrated initial baselines from live camera frame.")
        else:
            logger.warning("Camera failed to deliver valid initial frame. Marking camera OFFLINE.")
            with self.db_mgr.get_connection() as conn:
                for cam in self.cfg.cameras:
                    conn.execute("UPDATE cameras SET status = 'OFFLINE' WHERE id = ?", (cam.id,))
                conn.commit()

        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("TRACE Pipeline started successfully.")

    def stop(self) -> None:
        """Stops the processing pipeline."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self.capture.stop()
        logger.info("TRACE Pipeline stopped.")

    def _run_loop(self) -> None:
        target_fps = 5
        interval = 1.0 / target_fps
        t_last = time.time()

        while self.is_running:
            t_start = time.time()
            success, raw_frame, capture_t = self.capture.read_latest()

            if success and raw_frame is not None:
                # 1. Apply demo scenario perturbations if triggered
                injected_frame, synthetic_person_mask = self.demo_injector.apply_to_frame(
                    raw_frame, self.assets, self.empty_rois
                )
                # Sync outbox network state with demo injector
                self.outbox.set_network_state(not self.demo_injector.is_network_cut())

                # 2. Camera health & optical obstruction evaluation
                health_res = self.health_monitor.check(injected_frame, capture_t, landmark_ratio=self.aligner.inlier_ratio)
                camera_ok = health_res["ok"] and not health_res["is_offline"]

                # 3. Environmental alignment & camera tampering check
                align_res = self.aligner.align_frame(injected_frame)
                is_cam_tampered = align_res.get("is_tampered", False)

                # 4. People segmentation & privacy occlusion mask
                person_mask = self.people_segmenter.segment(injected_frame, extra_mask=synthetic_person_mask)

                # 5. Process registered assets
                annotated = injected_frame.copy()
                now = time.time()

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

                # 7. Flush outbox store-and-forward queue
                self.outbox.flush()

                # 8. Render HUD overlays on annotated frame
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
                self.health_monitor.check(None, capture_t)
                with self.db_mgr.get_connection() as conn:
                    for cam in self.cfg.cameras:
                        conn.execute("UPDATE cameras SET status = 'OFFLINE' WHERE id = ?", (cam.id,))
                    conn.commit()

            # Maintain target cadence
            elapsed = time.time() - t_start
            sleep_s = interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

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

        # Transform polygon by camera homography drift if applicable
        aligned_region = self.aligner.transform_polygon(raw_region)
        pts = np.array(aligned_region, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        h, w = frame_bgr.shape[:2]

        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))
        rw = max(1, min(w - rx, rw))
        rh = max(1, min(h - ry, rh))

        live_crop = frame_bgr[ry : ry + rh, rx : rx + rw]

        # Visibility and occlusion calculation
        occlusion_ratio, visibility = self.people_segmenter.calculate_visibility_and_occlusion(
            person_mask=person_mask,
            region_pts=aligned_region,
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

        # Asset localization and physical displacement
        found_elsewhere = False
        asset_displacement_frac = 0.0
        relocated_score = 0.0

        # When presence in expected region drops, search wider frame to distinguish relocation from missing
        if base_median is not None and ver_res["present_score"] < asset_cfg.thresholds.present:
            found_elsewhere, relocated_bbox, relocated_score = self.relocator.search(
                frame_bgr=frame_bgr,
                template_crop_bgr=base_median,
                exclude_zone=asset_cfg.zone or raw_region,
                person_mask=person_mask,
            )
            if found_elsewhere and relocated_bbox is not None:
                bx, by, bw, bh = relocated_bbox
                obs_region = [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]
                _, asset_displacement_frac = self.verifier.calculate_centroid_drift(obs_region, raw_region)

        # State change prediction (e.g. Safe door open/closed)
        state_label, state_conf = self.state_classifier.predict(asset_id, live_crop)

        # Build Observation with genuine asset displacement
        obs = Obs(
            t=timestamp,
            present_score=ver_res["present_score"],
            drift_frac=asset_displacement_frac,
            visibility=visibility,
            empty_match=ver_res["empty_match"],
            camera_ok=camera_ok and not is_cam_tampered,
            found_elsewhere=found_elsewhere,
            state_label=state_label,
            state_conf=state_conf,
            tag_matched=ver_res["tag_matched"],
        )

        # Step State Machine
        prev_state = fsm.state.value
        new_state = fsm.step(obs).value

        # Calculate event-specific confidence (distinct from generic presence score)
        if new_state == "MOVED":
            event_confidence = max(0.75, float(relocated_score))
        elif new_state == "MISSING":
            event_confidence = round(float(min(1.0, (1.0 - ver_res["present_score"]) * visibility)), 2)
        elif new_state == "SWAPPED":
            event_confidence = 0.95
        else:
            event_confidence = ver_res["present_score"]

        # Update database `states` table
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

        # Check Policy Engine for Alert Dispatch
        if new_state != prev_state and new_state not in ("SAFE", "OCCLUDED", "RETURNED"):
            self._handle_alert_transition(
                asset_id=asset_id,
                asset_cfg=asset_cfg,
                prev_state=prev_state,
                new_state=new_state,
                confidence=event_confidence,
                details={
                    "drift_frac": round(asset_displacement_frac, 3),
                    "threshold": asset_cfg.thresholds.move_frac,
                    "visibility": round(visibility, 2),
                    "empty_match": round(ver_res["empty_match"], 2),
                    "camera_ok": camera_ok,
                },
                live_crop=live_crop,
                base_crop=base_median if base_median is not None else live_crop,
                person_mask_crop=person_mask[ry : ry + rh, rx : rx + rw],
                timestamp=timestamp,
            )

        # Draw asset box on annotated frame
        self._draw_asset_box(annotated_frame, pts, asset_cfg.name, new_state, fsm.belief)

    def _handle_alert_transition(
        self,
        asset_id: str,
        asset_cfg: AssetConfig,
        prev_state: str,
        new_state: str,
        confidence: float,
        details: Dict[str, Any],
        live_crop: np.ndarray,
        base_crop: np.ndarray,
        person_mask_crop: np.ndarray,
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
        ev_meta, crop_hashes = self.evidence_writer.save_crops(
            event_id=event_id,
            before_crop=base_crop,
            after_crop=live_crop,
            before_mask=person_mask_crop if self.cfg.privacy.mask_people else None,
            after_mask=person_mask_crop if self.cfg.privacy.mask_people else None,
            person_masked=self.cfg.privacy.mask_people,
        )

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

        # Atomically commit ledger append, incidents, events, and outbox enqueue in one transaction
        with self.db_mgr.get_connection() as conn:
            try:
                ledger_info = self.ledger_mgr.append(
                    db=conn,
                    event=event_payload,
                    crop_hashes=crop_hashes,
                    commit=False,
                )
                event_payload["ledger"] = ledger_info

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

                # Atomically enqueue into outbox queue
                self.outbox.enqueue_conn(conn, event_id, json.dumps(event_payload))
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Atomic alert persistence failed for {event_id}: {e}")
                raise

        # Record for deduplication
        self.policy_engine.record_alert_dispatched(asset_id, new_state, incident_id, timestamp)

        # Dispatch alerts if not suppressed
        if not policy_res.get("suppress_push"):
            self.alert_router.dispatch(event_payload)
            logger.info(f"SECURITY ALERT [{severity.upper()}] {asset_id}: {new_state} (conf {confidence:.2f})")

    def _handle_new_object(
        self,
        blob_info: Dict[str, Any],
        frame_bgr: np.ndarray,
        person_mask: np.ndarray,
        timestamp: float
    ) -> None:
        bx, by, bw, bh = blob_info["bbox"]
        h, w = frame_bgr.shape[:2]
        crop = frame_bgr[max(0, by): min(h, by+bh), max(0, bx): min(w, bx+bw)]
        if crop.size == 0:
            return

        event_id = str(uuid.uuid4())
        dt_str = datetime.datetime.fromtimestamp(timestamp).astimezone().isoformat()

        # Evidence crop
        ev_meta, crop_hashes = self.evidence_writer.save_crops(
            event_id=event_id,
            before_crop=np.full_like(crop, 180),
            after_crop=crop,
            person_masked=True,
        )

        event_payload = {
            "id": event_id,
            "asset_id": blob_info["id"],
            "camera_id": "cam-1",
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

        with self.db_mgr.get_connection() as conn:
            # Register discovered new object in assets and states tables
            conn.execute(
                """
                INSERT OR IGNORE INTO assets (
                    id, name, class, criticality, policy_mode, camera_id,
                    region_polygon, zone_polygon, tag_id, present_threshold, active
                ) VALUES (?, ?, 'new_object', 2, 'fixed', 'cam-1', '[]', '[]', NULL, 0.70, 1)
                """,
                (blob_info["id"], f"Unregistered Object ({blob_info['id']})")
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO states (
                    asset_id, state, belief, confidence, last_verified_at, updated_at
                ) VALUES (?, 'NEW_OBJECT', 0.91, 0.91, ?, ?)
                """,
                (blob_info["id"], timestamp, timestamp)
            )

            ledger_info = self.ledger_mgr.append(conn, event_payload, crop_hashes)
            event_payload["ledger"] = ledger_info
            incident_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO incidents (id, asset_id, opened_at, status, counter) VALUES (?, ?, ?, 'OPEN', 1)",
                (incident_id, blob_info["id"], timestamp)
            )
            conn.execute(
                """
                INSERT INTO events (
                    id, asset_id, camera_id, type, severity, confidence, status,
                    detected_at, received_at, before_path, after_path, payload_json, incident_id
                ) VALUES (?, ?, 'cam-1', 'NEW_OBJECT', 'medium', 0.91, 'NEW', ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id, blob_info["id"], timestamp, timestamp,
                    ev_meta["before"], ev_meta["after"], json.dumps(event_payload), incident_id
                )
            )
            conn.commit()

        self.outbox.enqueue(event_id, json.dumps(event_payload))
        self.alert_router.dispatch(event_payload)

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
