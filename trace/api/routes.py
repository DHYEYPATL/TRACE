"""
FastAPI Routes module for TRACE REST and Streaming endpoints.
Implements the complete API specification from Section 7.3.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trace.api.websocket import ConnectionManager
from trace.config import AssetConfig
from trace.pipeline import TracePipeline

logger = logging.getLogger("trace.api")


# Request Schemas
class RegisterAssetRequest(BaseModel):
    id: str
    name: str
    class_name: str = Field(default="safe", alias="class")
    criticality: int = 2
    mode: str = "fixed"
    camera_id: str = "cam-1"
    tag_id: Optional[int] = None
    region: List[List[int]]
    zone: Optional[List[List[int]]] = None
    present_threshold: float = 0.70


class AuthorizeMoveRequest(BaseModel):
    duration_minutes: int = 10
    actor: str = "Branch Manager"


class InjectFaultRequest(BaseModel):
    fault_type: str # cover_lens, dim_lights, occlude_asset, cut_network, move_asset, remove_asset, place_unknown_object, swap_asset, knock_camera, tamper_ledger, reset
    asset_id: Optional[str] = None
    duration_seconds: float = 6.0


class ActionRequest(BaseModel):
    actor: str = "Operator"
    note: str = ""


ALLOWED_EVENT_TYPES = {
    "MOVED", "MISSING", "SWAPPED", "NEW_OBJECT", "SAFE",
    "OCCLUDED", "RETURNED", "CAMERA_OBSTRUCTED", "CAMERA_OFFLINE",
    "CAMERA_TAMPERED", "UNKNOWN"
}


class IngestEventRequest(BaseModel):
    id: Optional[str] = None
    asset_id: str
    camera_id: str = "cam-1"
    type: str = "UNKNOWN"
    severity: str = "medium"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    detected_at: Optional[float] = None
    incident_id: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None


def get_auth_role(request: Request, pipeline: TracePipeline) -> str:
    """
    Extracts and validates credentials from Authorization header or X-API-Key.
    Query-string token transmission is strictly disallowed to prevent exposure in logs.
    Unauthenticated callers receive 'viewer' role when require_auth=False; never 'admin'.
    """
    auth_header = request.headers.get("Authorization")
    api_key_header = request.headers.get("X-API-Key")

    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    elif api_key_header:
        token = api_key_header.strip()

    sec = pipeline.cfg.security
    if token:
        if token == sec.admin_token:
            return "admin"
        elif token == sec.operator_token:
            return "operator"
        elif token == sec.viewer_token:
            return "viewer"
        else:
            raise HTTPException(status_code=401, detail="Invalid authentication token")

    if sec.require_auth:
        raise HTTPException(
            status_code=401,
            detail="Authentication required: missing token in Authorization or X-API-Key header"
        )

    # When authentication is not strictly required (demo/local mode), default to viewer (read-only)
    return "viewer"


def check_permission(request: Request, pipeline: TracePipeline, min_role: str = "viewer") -> str:
    """Enforces role-based access control (viewer < operator < admin)."""
    role = get_auth_role(request, pipeline)
    hierarchy = {"viewer": 1, "operator": 2, "admin": 3}
    if hierarchy.get(role, 0) < hierarchy.get(min_role, 1):
        raise HTTPException(status_code=403, detail=f"Forbidden: requires '{min_role}' role (current: '{role}')")
    return role


def validate_polygon_region(region: List[List[int]], max_w: int = 1920, max_h: int = 1080) -> None:
    """Validates polygon geometry and bounds."""
    if not region or len(region) < 3:
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 vertices")
    for pt in region:
        if len(pt) != 2:
            raise HTTPException(status_code=400, detail="Each vertex must be a 2D [x, y] coordinate")
        x, y = pt
        if not (0 <= x <= max_w and 0 <= y <= max_h):
            raise HTTPException(status_code=400, detail=f"Vertex [{x}, {y}] exceeds valid camera frame boundaries")


def create_app(pipeline: TracePipeline) -> FastAPI:
    app = FastAPI(
        title="TRACE - Asset Confidence Engine",
        description="Software-only, privacy-first asset confidence monitoring API",
        version="1.0.0"
    )

    ws_manager = ConnectionManager()

    # Register pipeline callback to broadcast alerts via websocket
    def _on_alert_callback(msg: Dict[str, Any]) -> None:
        asyncio.run(ws_manager.broadcast_json(msg))

    pipeline.alert_router.register_ws_callback(_on_alert_callback)

    # Mount static assets
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Secure Evidence Retrieval (requires viewer authorization and prevents directory traversal)
    @app.get("/api/evidence/{date_str}/{event_id}/{filename}")
    @app.get("/evidence/{date_str}/{event_id}/{filename}")
    def get_evidence_file(date_str: str, event_id: str, filename: str, request: Request):
        check_permission(request, pipeline, min_role="viewer")
        base_dir = Path("data/evidence").resolve()
        target_path = (base_dir / date_str / event_id / filename).resolve()
        if not str(target_path).startswith(str(base_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
        if not target_path.is_file():
            raise HTTPException(status_code=404, detail="Evidence file not found")
        return FileResponse(str(target_path))

    # Serve main Dashboard HTML at root
    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h2>TRACE Console Loading...</h2>")

    # 1. Asset Registration & Management
    @app.post("/api/assets")
    def register_asset(req: RegisterAssetRequest, request: Request):
        check_permission(request, pipeline, min_role="admin")
        validate_polygon_region(req.region)
        if req.zone:
            validate_polygon_region(req.zone)

        zone_pts = req.zone or req.region
        asset_cfg = AssetConfig(
            id=req.id,
            name=req.name,
            class_name=req.class_name,
            criticality=req.criticality,
            mode=req.mode,
            camera_id=req.camera_id,
            tag_id=req.tag_id,
            region=req.region,
            zone=zone_pts,
        )
        asset_cfg.thresholds.present = req.present_threshold

        pipeline.assets[req.id] = asset_cfg
        pipeline._init_assets_and_baselines()

        return {"ok": True, "asset_id": req.id, "message": "Asset registered successfully."}

    @app.get("/api/assets")
    def list_assets():
        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute("SELECT * FROM assets WHERE active = 1")
            rows = [dict(r) for r in cur.fetchall()]
        return {"ok": True, "assets": rows}

    @app.get("/api/assets/{asset_id}")
    def get_asset(asset_id: str):
        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Asset not found")
        return {"ok": True, "asset": dict(row)}

    @app.post("/api/assets/{asset_id}/baseline/accept")
    def accept_new_baseline(asset_id: str, request: Request):
        """Accepts current camera crop as the new official baseline location (e.g. after authorized move)."""
        check_permission(request, pipeline, min_role="operator")
        if asset_id not in pipeline.assets:
            raise HTTPException(status_code=404, detail="Asset not found")

        with pipeline.lock:
            frame = pipeline.latest_clean_frame
        if frame is None:
            raise HTTPException(status_code=500, detail="No active camera frame")

        a = pipeline.assets[asset_id]
        pts = np.array(a.region, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        crop = frame[ry:ry+rh, rx:rx+rw].copy()

        pipeline.baseline_medians[asset_id] = crop
        pipeline.baseline_variants[asset_id] = pipeline.verifier.extractor.generate_variants(crop)

        # Persist new accepted baseline to disk and SQLite so it survives restarts
        baselines_dir = Path("data/baselines")
        baselines_dir.mkdir(parents=True, exist_ok=True)
        img_path = baselines_dir / f"{asset_id}_accepted_{int(time.time())}.jpg"
        cv2.imwrite(str(img_path), crop)

        with pipeline.db_mgr.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO baselines (id, asset_id, kind, image_path, created_at)
                VALUES (?, ?, 'median', ?, ?)
                """,
                (f"base-{asset_id}-{int(time.time())}", asset_id, str(img_path).replace("\\", "/"), time.time())
            )
            conn.commit()

        # Reset FSM
        fsm = pipeline.fsms.get(asset_id)
        if fsm:
            fsm.state = pipeline.fsms[asset_id].cfg and pipeline.fsms[asset_id].state.__class__.SAFE
            fsm.doubt_engine.reset_sighted(time.time())

        return {"ok": True, "asset_id": asset_id, "message": "New baseline accepted and persisted."}

    @app.post("/api/assets/{asset_id}/baseline")
    def capture_baseline(asset_id: str, kind: str = "median", request: Request = None):
        """Captures a baseline crop of the asset (kind: 'median', 'empty', or 'variant')."""
        if request:
            check_permission(request, pipeline, min_role="operator")
        if asset_id not in pipeline.assets:
            raise HTTPException(status_code=404, detail="Asset not found")

        with pipeline.lock:
            frame = pipeline.latest_clean_frame
        if frame is None:
            raise HTTPException(status_code=500, detail="No active camera frame")

        a = pipeline.assets[asset_id]
        pts = np.array(a.region, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        crop = frame[ry:ry+rh, rx:rx+rw].copy()

        if kind == "empty":
            pipeline.empty_rois[asset_id] = crop
        else:
            pipeline.baseline_medians[asset_id] = crop
            pipeline.baseline_variants[asset_id] = pipeline.verifier.extractor.generate_variants(crop)

        return {"ok": True, "asset_id": asset_id, "kind": kind, "message": f"{kind.capitalize()} baseline captured."}

    @app.post("/api/assets/{asset_id}/authorize")
    def authorize_move(asset_id: str, req: AuthorizeMoveRequest, request: Request = None):
        if request:
            check_permission(request, pipeline, min_role="operator")
        res = pipeline.policy_engine.authorize_move(
            asset_id=asset_id,
            duration_minutes=req.duration_minutes,
            actor=req.actor
        )
        return res


    # Review Queue (Selective Prediction, Section 3.3 #5 & 8.5)
    @app.get("/api/review-queue")
    def get_review_queue():
        """Returns events with confidence 0.60 to 0.89 awaiting human review."""
        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute("SELECT * FROM events WHERE status = 'NEEDS_REVIEW' ORDER BY detected_at DESC")
            rows = [dict(r) for r in cur.fetchall()]
        return {"ok": True, "count": len(rows), "items": rows}

    @app.post("/api/review-queue/{event_id}/confirm")
    def confirm_review_event(event_id: str, req: ActionRequest, request: Request = None):
        """Promotes review event to an active security alert and dispatches notification."""
        if request:
            check_permission(request, pipeline, min_role="operator")
        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute("SELECT payload_json FROM events WHERE id = ?", (event_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Event not found")
            conn.execute("UPDATE events SET status = 'CONFIRMED' WHERE id = ?", (event_id,))
            conn.commit()

            p_obj = json.loads(row[0])
            p_obj["status"] = "CONFIRMED"
            pipeline.alert_router.dispatch(p_obj)

        return {"ok": True, "event_id": event_id, "status": "CONFIRMED"}

    @app.post("/api/review-queue/{event_id}/dismiss")
    def dismiss_review_event(event_id: str, req: ActionRequest, request: Request = None):
        """Dismisses review event, recording a hard negative for model calibration."""
        if request:
            check_permission(request, pipeline, min_role="operator")
        with pipeline.db_mgr.get_connection() as conn:
            conn.execute("UPDATE events SET status = 'DISMISSED' WHERE id = ?", (event_id,))
            conn.commit()
        return {"ok": True, "event_id": event_id, "status": "DISMISSED"}

    # Dynamic Settings & Policy Tuning (Screen 7, Section 8.2)
    @app.get("/api/settings")
    def get_settings():
        return {
            "site": pipeline.cfg.site,
            "profile": pipeline.cfg.profile,
            "business_hours": pipeline.cfg.business_hours.model_dump(),
            "timers": pipeline.cfg.get_timers().model_dump(),
            "half_lives": pipeline.cfg.half_lives_s.get(pipeline.cfg.profile, {}),
        }

    @app.post("/api/settings")
    def update_settings(payload: Dict[str, Any], request: Request):
        """Dynamically tunes timers, business hours, and thresholds without restarting."""
        check_permission(request, pipeline, min_role="admin")
        if "timers" in payload:
            t = pipeline.cfg.get_timers()
            for k, v in payload["timers"].items():
                if hasattr(t, k):
                    setattr(t, k, float(v))
        if "business_hours" in payload:
            bh = pipeline.cfg.business_hours
            if "start" in payload["business_hours"]:
                bh.start = str(payload["business_hours"]["start"])
            if "end" in payload["business_hours"]:
                bh.end = str(payload["business_hours"]["end"])
        return {"ok": True, "message": "Settings updated dynamically."}

    # Ingest API (Edge-to-Server idempotent sync, Section 7.3)
    @app.post("/api/ingest/events")
    def ingest_event(req: IngestEventRequest, request: Request):
        """Idempotent event ingest from edge nodes with strict schema and RBAC enforcement."""
        check_permission(request, pipeline, min_role="operator")
        if req.type not in ALLOWED_EVENT_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid event type '{req.type}'")

        ev_id = req.id or str(uuid.uuid4())
        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute("SELECT id FROM events WHERE id = ?", (ev_id,))
            if cur.fetchone():
                return {"ok": True, "status": "DUPLICATE_IGNORED", "id": ev_id}

            now = time.time()
            dt = req.detected_at if req.detected_at is not None else now
            ev_meta = req.evidence or {}
            event_payload = req.model_dump()
            event_payload["source"] = "EXTERNAL_INGEST"

            conn.execute(
                """
                INSERT INTO events (
                    id, asset_id, camera_id, type, severity, confidence, status,
                    detected_at, received_at, before_path, after_path, payload_json, incident_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'INGESTED', ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev_id, req.asset_id, req.camera_id,
                    req.type, req.severity,
                    req.confidence, dt, now,
                    ev_meta.get("before"),
                    ev_meta.get("after"),
                    json.dumps(event_payload), req.incident_id
                )
            )
            conn.commit()
        return {"ok": True, "status": "INGESTED", "id": ev_id}

    # 2. State & Live Metrics
    @app.get("/api/state")
    def get_current_states():
        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute("""
                SELECT a.id, a.name, a.class, a.criticality, a.policy_mode,
                       s.state, s.belief, s.confidence, s.last_verified_at, s.updated_at
                FROM assets a
                LEFT JOIN states s ON a.id = s.asset_id
                WHERE a.active = 1
            """)
            assets_status = [dict(r) for r in cur.fetchall()]

        return {
            "ok": True,
            "site": pipeline.cfg.site,
            "timestamp": time.time(),
            "assets": assets_status,
            "fps": round(pipeline.current_fps, 1),
            "camera_status": pipeline.health_monitor.is_obstructed and "OBSTRUCTED" or "HEALTHY",
            "network_status": pipeline.outbox.network_connected and "ONLINE" or "OFFLINE",
            "pending_outbox": pipeline.outbox.get_pending_count(),
        }

    # 3. Events & Incidents
    @app.get("/api/events")
    def list_events(
        asset_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50
    ):
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if asset_id:
            query += " AND asset_id = ?"
            params.append(asset_id)
        if event_type:
            query += " AND type = ?"
            params.append(event_type)

        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]

        return {"ok": True, "count": len(rows), "events": rows}

    @app.get("/api/events/{event_id}")
    def get_event(event_id: str):
        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            ev = cur.fetchone()
            if not ev:
                raise HTTPException(status_code=404, detail="Event not found")
            ev_dict = dict(ev)

            # Fetch associated actions
            cur_act = conn.execute("SELECT * FROM actions WHERE event_id = ? ORDER BY at ASC", (event_id,))
            actions = [dict(a) for a in cur_act.fetchall()]
            ev_dict["actions"] = actions

        return {"ok": True, "event": ev_dict}

    @app.post("/api/events/{event_id}/{action_name}")
    def handle_event_action(event_id: str, action_name: str, req: ActionRequest, request: Request = None):
        if request:
            check_permission(request, pipeline, min_role="operator")
        if action_name not in ("ack", "resolve", "false_positive"):
            raise HTTPException(status_code=400, detail="Invalid action")

        now = time.time()
        action_id = str(uuid.uuid4())
        action_map = {
            "ack": "ACKNOWLEDGED",
            "resolve": "RESOLVED",
            "false_positive": "FALSE_POSITIVE"
        }
        status_val = action_map[action_name]

        with pipeline.db_mgr.get_connection() as conn:
            # 1. Update event status
            conn.execute("UPDATE events SET status = ? WHERE id = ?", (status_val, event_id))

            # 2. Append to actions table
            conn.execute(
                "INSERT INTO actions (id, event_id, actor, action, note, at) VALUES (?, ?, ?, ?, ?, ?)",
                (action_id, event_id, req.actor, action_name, req.note, now)
            )

            # 3. Log action to cryptographic ledger
            action_event_payload = {
                "id": action_id,
                "event_id": event_id,
                "type": f"ACTION_{action_name.upper()}",
                "actor": req.actor,
                "note": req.note,
                "at": now,
            }
            pipeline.ledger_mgr.append(conn, action_event_payload)
            conn.commit()

        return {"ok": True, "event_id": event_id, "action": action_name, "status": status_val}

    # 4. Cryptographic Ledger & Audit
    @app.get("/api/ledger/verify")
    def verify_ledger():
        with pipeline.db_mgr.get_connection() as conn:
            res = pipeline.ledger_mgr.verify(conn, verify_disk_crops=True)
        return res

    @app.get("/api/ledger/head")
    def get_ledger_head():
        with pipeline.db_mgr.get_connection() as conn:
            cur = conn.execute("SELECT idx, hash, sig FROM ledger ORDER BY idx DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                return {"idx": 0, "hash": "0" * 64, "sig": ""}
            return {"idx": row[0], "hash": row[1], "sig": row[2]}

    # 5. System Health & Observability
    @app.get("/api/health")
    def get_system_health():
        meta = pipeline.verifier.get_model_metadata()
        return {
            "status": "healthy" if not pipeline.health_monitor.is_obstructed else "degraded",
            "camera_obstructed": pipeline.health_monitor.is_obstructed,
            "camera_offline": pipeline.health_monitor.is_offline,
            "fps": round(pipeline.current_fps, 1),
            "queue_depth": pipeline.outbox.get_pending_count(),
            "network_connected": pipeline.outbox.network_connected,
            "total_frames_processed": pipeline.total_frames_processed,
            "model_version": meta["model_version"],
            "model_metadata": meta,
        }

    @app.get("/metrics")
    def get_metrics():
        return {
            "trace_fps": pipeline.current_fps,
            "trace_queue_depth": pipeline.outbox.get_pending_count(),
            "trace_frames_total": pipeline.total_frames_processed,
            "trace_assets_active": len(pipeline.assets),
            "trace_network_connected": 1 if pipeline.outbox.network_connected else 0,
        }

    # 6. Demo Scenario Fault Injector
    @app.post("/api/demo/inject")
    def inject_demo_fault(req: InjectFaultRequest, request: Request):
        if not pipeline.cfg.security.demo_mode:
            raise HTTPException(
                status_code=403,
                detail="Demo fault injection is disabled. Set security.demo_mode=true in config to enable."
            )
        check_permission(request, pipeline, min_role="admin")

        if req.fault_type == "reset":
            return pipeline.demo_injector.reset_all()

        if req.fault_type == "tamper_ledger":
            with pipeline.db_mgr.get_connection() as conn:
                res = pipeline.ledger_mgr.inject_tamper_for_demo(conn)
            return res

        res = pipeline.demo_injector.trigger_fault(
            fault_type=req.fault_type,
            asset_id=req.asset_id,
            duration_s=req.duration_seconds
        )
        return res

    # 7. Live MJPEG Video Stream
    @app.get("/api/video/feed")
    def get_video_feed():
        def _mjpeg_generator():
            while True:
                with pipeline.lock:
                    frame = pipeline.latest_annotated_frame
                if frame is not None:
                    ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                        )
                time.sleep(0.08) # ~12 fps preview stream

        return StreamingResponse(
            _mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    # 8. WebSocket Stream
    @app.websocket("/ws/stream")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                # Periodic state heartbeat
                with pipeline.db_mgr.get_connection() as conn:
                    cur = conn.execute("""
                        SELECT a.id, a.name, s.state, s.belief, s.confidence
                        FROM assets a LEFT JOIN states s ON a.id = s.asset_id
                    """)
                    states_list = [dict(r) for r in cur.fetchall()]

                msg = {
                    "kind": "state_heartbeat",
                    "fps": round(pipeline.current_fps, 1),
                    "camera_status": "OBSTRUCTED" if pipeline.health_monitor.is_obstructed else "HEALTHY",
                    "network_status": "ONLINE" if pipeline.outbox.network_connected else "OFFLINE",
                    "queue_depth": pipeline.outbox.get_pending_count(),
                    "assets": states_list,
                    "timestamp": time.time(),
                }
                await websocket.send_text(json.dumps(msg))
                await asyncio.sleep(0.5) # 2 Hz update
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception:
            await ws_manager.disconnect(websocket)

    return app
