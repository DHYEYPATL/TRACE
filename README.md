# AssetWatch AI — Enterprise Visual Asset Monitoring

Local-first, real-time visual asset monitoring from a USB webcam on a single laptop. No cloud, no tags, no pre-labelling. Register objects once; the system memorises them and alerts on movement or absence.

## What's in this build (v1.0)

### Core detection & tracking
- **YOLO11n + BoT-SORT** — automatic multi-object detection at 8 fps target on GPU
- **USB webcam** with DirectShow/MSMF fallback, auto-reconnect, separate camera/inference threads
- **CSRT manual tracker** for objects the detector doesn't recognise — just draw a box
- **CSRT → YOLO promotion** — when AI detection overlaps a manual tracker with matching appearance, automatically upgrades to AI tracking

### Viewpoint-robust identity (enterprise upgrade)
- **10-sample gallery** per asset (up from 5) — captures front, ±45°, side, top-down
- **Online gallery enrichment** — while tracked at high confidence (≥ 0.85), diverse new views are added automatically; full gallery is not needed at registration time
- **Partial-view adaptive weighting** — when detected box < 60% of last known area, shifts weight from deep embedding toward colour + spatial cues
- **Two-stage matching** — rolling short-term buffer (last 6 frames, priority for lost assets) + persistent gallery with spatial Gaussian prior
- **ResNet18 (512-d) + HSV histogram + ORB keypoints** composite score; never guesses when ambiguous

### State machine & alerts
- States: `PRESENT` → `CHECKING` (3 s) → `MISSING` (25 s) / `MOVED` / `UNVERIFIED`
- Camera health gate — no MISSING decision while camera is disconnected or stale
- Recovery detection — `↩ Recovered` badge + cyan box + 880 Hz tone when a lost asset is re-identified
- Audio tones: 520 Hz = missing, 760 Hz = moved, 880 Hz = recovered

### Operations
- **Monitoring modes**: Always / Day (08:00–20:00) / Office / Night — idle drops to 2 fps, saves GPU heat
- **Mode persistence** — last set mode stored in SQLite, restored on restart
- **Hard delete** — removes asset, all events, and crop files from disk (`POST /api/delete/<id>`)
- **Soft archive** — keeps history, removes from monitoring

### Storage (enterprise minimalism)
| Item | Limit |
|---|---|
| Gallery samples per asset | 10 |
| Rolling embeddings (RAM) | 6 per asset |
| Reference crop thumbnails on disk | 3 per asset (max 160 px, JPEG Q=75) |
| Total disk for 50 assets | ≈ 2–3 MB |
| RAM for identity cache (100 assets) | < 15 MB |

---

## Quick start — Windows 10/11

1. Plug in the USB webcam.
2. Double-click `Setup-AssetWatch.bat` and wait.
3. Double-click `Run-AssetWatch.bat`.
4. Open **http://127.0.0.1:5055** in your browser.
5. Camera 1 (USB webcam) is selected by default. If the laptop camera appears, try index 0 or 2 and click **Reconnect**.
6. The intro guide opens on first launch — read it, then close it.
7. Click a detected amber box → name it → **Register selected item**.
8. Add 4–6 reference views using **＋ Reference view** from different angles. Watch the coverage meter fill green.

First startup downloads YOLO (~5 MB) and ResNet18 (~45 MB) weights — needs internet once.

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/video_feed` | MJPEG stream |
| GET | `/api/state` | Full state (runtime, assets, events) |
| POST | `/api/select` | Select object by click or drag |
| POST | `/api/register` | Register selected object |
| POST | `/api/reference/<id>` | Add reference view |
| POST | `/api/archive/<id>` | Soft archive (keeps history) |
| POST | `/api/delete/<id>` | **Hard delete** (removes all data) |
| POST | `/api/ack/<event_id>` | Acknowledge event |
| POST | `/api/camera` | Switch camera index |
| POST | `/api/mode` | Set monitoring mode |

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `ASSETWATCH_CAMERA_INDEX` | `1` | Default camera (1 = USB webcam) |
| `ASSETWATCH_INFER_FPS` | `8.0` | Target inference rate |
| `ASSETWATCH_MISSING_AFTER` | `25.0` | Seconds before MISSING |
| `ASSETWATCH_MOVEMENT_FRACTION` | `0.11` | Movement threshold (frame diagonal %) |
| `ASSETWATCH_MATCH_THRESHOLD` | `0.76` | Min similarity to assign identity |
| `ASSETWATCH_MATCH_MARGIN` | `0.07` | Min gap over 2nd candidate |
| `ASSETWATCH_GALLERY_MAX` | `10` | Gallery samples per asset |
| `ASSETWATCH_ENRICH_MIN_CONF` | `0.85` | Min confidence for auto-enrichment |
| `ASSETWATCH_ENRICH_DIVERSITY` | `0.08` | Min cosine distance for new gallery sample |
| `ASSETWATCH_MODE` | `always` | Monitoring mode (always/day/office/night) |
| `ASSETWATCH_DAY_START` | `08:00` | Day/office mode start |
| `ASSETWATCH_DAY_END` | `20:00` | Day/office mode end |
| `ASSETWATCH_IDLE_FPS` | `2.0` | Inference rate in idle mode |
| `ASSETWATCH_MODEL` | `yolo11n.pt` | YOLO model name |
| `ASSETWATCH_IMGSZ` | `640` | YOLO inference resolution |
| `ASSETWATCH_CONFIDENCE` | `0.25` | YOLO detection confidence |
| `ASSETWATCH_PORT` | `5055` | Flask port |
| `ASSETWATCH_W_DEEP` | `0.55` | Deep embedding weight in composite score |
| `ASSETWATCH_W_COLOR` | `0.20` | HSV histogram weight |
| `ASSETWATCH_W_ORB` | `0.15` | ORB keypoint weight |
| `ASSETWATCH_W_SPATIAL` | `0.10` | Spatial prior weight |

---

## Project layout

```
app.py                        Flask app + all API endpoints
assetwatch/
  storage.py                  SQLite (assets, events, settings) + hard delete
  identity.py                 IdentityCache, IdentityState, enrichment logic
  vision.py                   YOLO detect, ResNet18 embed, composite scoring
  service.py                  Camera threads, inference loop, state machine
templates/index.html          Dashboard + intro modal
static/app.js                 Dashboard JS (coverage meter, mode toggle, hard delete)
static/app.css                Dashboard CSS
scripts/
  setup_windows.ps1           Setup (PyTorch CUDA, deps, unit tests)
  run_windows.ps1             Launcher
  check_setup.py              Full diagnostics (identity layer + monitoring mode)
tests/
  test_identity.py            Identity cache + enrichment + monitoring mode (24 tests)
  test_storage.py             Storage + hard delete + settings (8 tests)
  test_matching.py            Vision scoring (4 tests)
  test_service_states.py      State machine (3 tests)
data/                         SQLite db + reference thumbnails (created on first run)
docs/REAL_WORLD_VALIDATION.md Physical test checklist
```

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
# 39 passed
```

Covers: identity lifecycle, enrichment gates, plausible-location checks, monitoring modes, hard delete, settings, gallery limit, composite scoring, state machine, camera-health gate.

---

## Known limits

- Two visually identical, unmarked objects of the same class cannot be reliably distinguished when there is no visible difference — the system rejects ambiguous matches rather than guessing
- CSRT loses the object after full occlusion or leaving the frame; recovery needs manual re-selection or a matching YOLO detection nearby
- Movement alerts are relative to the registered anchor — re-register after moving the camera
- First startup needs internet to download YOLO (~5 MB) and ResNet18 (~45 MB) weights
- Local laptop monitoring only, not a certified security system
