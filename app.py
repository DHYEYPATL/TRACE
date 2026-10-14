from __future__ import annotations

import atexit
import os
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

from assetwatch.service import CameraService
from assetwatch.storage import Store

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ASSETWATCH_DATA_DIR", str(BASE_DIR / "data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
store   = Store(DATA_DIR / "assetwatch.sqlite3")
service = CameraService(store, DATA_DIR)

# Restore last-saved monitoring mode.
saved_mode = store.get_setting("mode", "always")
if saved_mode in ("always", "day", "office", "night"):
    import assetwatch.service as _svc
    _svc._MODE = saved_mode

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------
# Video
# ------------------------------------------------------------------

@app.get("/video_feed")
def video_feed():
    return Response(
        service.mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                 "Pragma": "no-cache"})


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

@app.get("/api/state")
def api_state():
    return jsonify(service.state())


# ------------------------------------------------------------------
# Selection & registration
# ------------------------------------------------------------------

@app.post("/api/select")
def api_select():
    try:
        data = request.get_json(force=True) or {}
        return jsonify({"ok": True, "selection": service.select_region(data)})
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/register")
def api_register():
    try:
        data  = request.get_json(force=True) or {}
        asset = service.register_selected(
            str(data.get("name", "")),
            str(data.get("category", "") or ""))
        return jsonify({"ok": True, "asset": _public_asset(asset)})
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Asset registration failed")
        return jsonify({"ok": False,
                        "error": f"Registration failed: {type(exc).__name__}: {exc}"}), 500


@app.post("/api/reference/<asset_id>")
def api_add_reference(asset_id: str):
    try:
        asset = service.add_reference_from_selection(asset_id)
        return jsonify({"ok": True, "asset": _public_asset(asset)})
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Reference capture failed")
        return jsonify({"ok": False,
                        "error": f"Reference capture failed: {type(exc).__name__}: {exc}"}), 500


# ------------------------------------------------------------------
# Asset lifecycle
# ------------------------------------------------------------------

@app.post("/api/archive/<asset_id>")
def api_archive(asset_id: str):
    if service.archive_asset(asset_id):
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Asset not found"}), 404


@app.post("/api/delete/<asset_id>")
def api_hard_delete(asset_id: str):
    """Hard delete — removes asset, all events, and crop files from disk."""
    if service.hard_delete_asset(asset_id):
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Asset not found"}), 404


# ------------------------------------------------------------------
# Events
# ------------------------------------------------------------------

@app.post("/api/ack/<int:event_id>")
def api_ack(event_id: int):
    return jsonify({"ok": store.acknowledge_event(event_id)})


# ------------------------------------------------------------------
# Camera
# ------------------------------------------------------------------

@app.post("/api/camera")
def api_camera():
    try:
        data  = request.get_json(force=True) or {}
        index = int(data.get("index", 1))
        service.change_camera(index)
        return jsonify({"ok": True, "camera_index": index})
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


# ------------------------------------------------------------------
# Monitoring mode
# ------------------------------------------------------------------

@app.post("/api/mode")
def api_set_mode():
    try:
        data = request.get_json(force=True) or {}
        service.set_mode(str(data.get("mode", "always")))
        return jsonify({"ok": True, "mode": data.get("mode")})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/debug/identity")
def api_debug_identity():
    """Dump the live IdentityCache — useful for threshold tuning."""
    out = []
    for s in service._identity_cache.all_states():
        out.append({
            "asset_id":            s.asset_id,
            "last_track_id":       s.last_track_id,
            "confidence":          round(s.confidence, 3),
            "consecutive_matches": s.consecutive_matches,
            "gallery_size":        s.gallery_size,
            "rolling_buffer_len":  len(s.rolling_embeddings),
            "lost_since":          s.lost_since,
            "seconds_since_seen":  round(service._identity_cache.seconds_since_seen(s.asset_id), 1),
            "recovered_from_gallery": s.recovered_from_gallery,
            "pending_enrich":      s.pending_enrich_embedding is not None,
        })
    return jsonify({"ok": True, "identity_cache": out, "count": len(out)})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _public_asset(asset: dict) -> dict:
    keys = ("asset_id", "name", "category", "tracker_mode", "status", "last_seen",
            "last_similarity", "active", "created_at", "updated_at", "anchor_box", "last_box")
    return {key: asset.get(key) for key in keys}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    service.start()
    atexit.register(service.stop)
    print("\nAssetWatch AI is starting. Open http://127.0.0.1:5055 in your browser.\n")
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("ASSETWATCH_PORT", "5055")),
        threaded=True, debug=False, use_reloader=False)
