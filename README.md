# TRACE: Trusted Real-time Asset Confidence Engine
### L&T Finance × Tech Tatva | Project IntelliWatch

> **"Other systems ask *'did it move?'*. TRACE asks *'how sure are we about every asset, right now?'*, and only alerts on positive evidence."**

---

## 1. System Design & Architectural Blueprint

TRACE is an enterprise software-only asset confidence engine built according to the **IntelliWatch System Blueprint** and system designs:

```
+---------------------------------------------------------------------------------------------------+
|                                            EDGE NODE                                              |
|                                                                                                   |
|  [Input Stream]                                                                                   |
|  Clips, Webcam, RTSP, or Synthetic Simulator                                                      |
|         │                                                                                         |
|         ▼                                                                                         |
|  [Health + Align] ──────► [People Mask] ──────► [Change Gate] ──────► [Region Verify]             |
|  Blur, Brightness,        Privacy blur          Fast CLAHE + SSIM      Fused Visual DNA 384-D     |
|  Texture, ArUco           in-memory RAM         (< 3 ms per asset)     + SIFT/ORB RANSAC inliers  |
|         │                                                                                         |
|         ▼                                                                                         |
|  [The Doubt Engine & FSM] ◄───────────────────────────────────────────────────────────────────────┘
|  Decaying confidence: C(t) = C0 * exp(-lambda * dt)                                               |
|  States: SAFE, OCCLUDED, UNKNOWN, MOVED, SWAPPED, STATE_CHANGED, MISSING, NEW_OBJECT, TAMPERED      |
|         │                                                                                         |
|         ▼                                                                                         |
|  [Policy Engine] ───────► [Signed Evidence Ledger] ─────► [Outbox Store-and-Forward]              |
|  Mobility schedules,      Ed25519 signature +             Offline resilience,                     |
|  Authorized moves         SHA-256 hash chaining           zero event loss                         |
+────────────────────────────────────────┬──────────────────────────────────────────────────────────+
                                         │ WebSocket & REST Sync
                                         ▼
+---------------------------------------------------------------------------------------------------+
|                                       SERVER & DASHBOARD                                          |
|                                                                                                   |
|  [FastAPI & WebSocket] ─► [Alert Router] ──────► [Telegram / ntfy / Console Audio Banner]        |
|  [SQLite WAL Mode]     ─► [Operations Console] ─► [Live MJPEG Feed + HUD Overlays]               |
|                           [Evidence Viewer]    ─► [Before & After Image Comparison Slider]        |
|                           [Ledger Audit]       ─► [Real-time Tamper Verification (Green/Red)]     |
|                           [Scenario Injector]  ─► [10 Live Round 2 Twist Fault Buttons]          |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Core Pillars & SOLID Engineering

### 1. The Positive-Evidence Rule (Demand 5)
Traditional vision analytics trigger theft alarms whenever an asset is covered by an operator or camera view is lost. In TRACE:
- `MISSING` is raised **only** when the asset's region is visible ($v \ge 0.85$), the camera is healthy, and the region matches the empty baseline ($empty\_match \ge 0.85$) for the configured wait time $T_2$.
- *"Can't see it"* is **never** *"it's gone"*.

### 2. The Doubt Engine (Demand 7 & Section 6.3)
Every asset maintains an active continuous belief confidence $C(t) \in [0, 1]$:
$$C(t) = C_0 \, e^{-\lambda_c \, (t - t_{seen})}, \qquad \lambda_c = \frac{\ln 2}{T_{half,c}}$$
- Verified sighting resets $C$ to 1.0.
- When $C(t) < 0.50$, state transitions to `UNKNOWN`.
- During human occlusion, decay rate is reduced to $0.25\times$. When camera is blind or offline, decays at full rate.

### 3. Cryptographic Tamper-Evident Ledger (Demand 8 & Section 10.2)
- Each event entry is serialized into canonical JSON, signed with **Ed25519** (`PyNaCl`), and linked into a **SHA-256** hash chain:
  $$h_n = \text{SHA-256}\Big(\text{canon}\big(\{\text{event}, \text{crops}, \text{prev}: h_{n-1}, \text{ts\_ns}\}\big)\Big)$$
- Database triggers (`ledger_no_update`, `ledger_no_delete`) reject any in-database modification.
- Live tamper verification verifies body hashes, chain continuity, Ed25519 signatures, and disk evidence crop hashes.

### 4. Privacy by Design (Section 5.4)
- Individuals (staff and customers) are segmented in volatile memory and masked using Gaussian blur ($\sigma=51$) before saving crops.
- Zero face recognition. Zero face storage. No raw unmasked footage stored at rest.

### 5. Policy Engine & Fatigue Mitigation (Demand 9 & Section 6.4)
- **Mobility rules:** Fixed assets alert on any move; mobile assets (laptops, chairs) are logged without push alerts during business hours.
- **Authorized Moves:** Operators can authorize moves for $N$ minutes (e.g. `/authorize printer-01 10`), suppressing alerts.
- **Deduplication:** Repeats within 5 minutes fold into the existing incident counter.

---

## 3. Directory Layout

```
TRACE/
├── config/
│   ├── demo.yaml             # Demo branch configuration (half-lives, timers, thresholds)
│   └── production.yaml       # Production branch configuration (RTSP, MinIO, MQTT)
├── trace/
│   ├── config.py             # Pydantic schema validation & environment variable resolution
│   ├── main.py               # Single-command launcher (Vision Pipeline + FastAPI + Dashboard)
│   ├── pipeline.py           # Core edge frame loop
│   ├── vision/
│   │   ├── capture.py        # Frame capture (Webcam, Synthetic simulator, RTSP, Video file)
│   │   ├── health.py         # Sharpness, brightness, texture, frozen feed monitor
│   │   ├── align.py          # ArUco/ORB homography, camera drift & TAMPERED detection
│   │   ├── people.py         # Temporal human segmentation & privacy masking
│   │   ├── gate.py           # CLAHE + SSIM + Delta-E fast change gate
│   │   ├── embed.py          # Visual DNA 384-D dense feature extractor
│   │   ├── verify.py         # Embedding cosine + SIFT/ORB RANSAC inlier fusion
│   │   ├── newobject.py      # Zone background model & persistent foreground blob tracker
│   │   └── stateclf.py       # Few-shot state classifier (e.g. Vault Door Open vs Closed)
│   ├── logic/
│   │   ├── belief.py         # Doubt Engine exponential decay model
│   │   ├── fsm.py            # AssetFSM (SAFE, OCCLUDED, UNKNOWN, MOVED, SWAPPED, etc.)
│   │   └── policy.py         # Mobility schedules, authorized moves, dedupe, escalation
│   ├── platform/
│   │   ├── db.py             # SQLite WAL mode, 13 tables, append-only triggers
│   │   ├── ledger.py         # Ed25519 signatures, SHA-256 hash chaining, audit verification
│   │   ├── evidence.py       # Masked before/after crop extraction & hashing
│   │   ├── outbox.py         # Store-and-forward queue & network-loss resilience
│   │   ├── alerts.py         # Telegram Bot API, ntfy, and WebSocket router
│   │   └── demo_inject.py    # 10 scenario edge-case fault injectors
│   ├── api/
│   │   ├── routes.py         # FastAPI REST endpoints & MJPEG live video feed
│   │   ├── websocket.py      # Real-time WebSocket connection manager
│   │   └── static/           # Enterprise Web Dashboard (HTML5, CSS3, JS, Live Canvas)
│   └── eval/
│       ├── metrics.py        # Wilson 95% confidence intervals, confusion matrix, P50/P95 latency
│       ├── baseline_naive.py # 5-tier ablation models
│       └── run_eval.py       # 720-trial automated evaluation harness
├── tests/
│   ├── test_ledger.py        # Cryptographic tamper, deleted row, crop hash tests
│   ├── test_invariants.py    # Blind camera override, positive evidence, privacy tests
│   ├── test_fsm.py           # Doubt Engine decay, 5-vote hysteresis, tag mismatch tests
│   └── test_policy.py        # Authorized move suppression, deduplication tests
├── CREDITS.md                # Open source credits & licences
└── pyproject.toml            # Project packaging specification
```

---

## 4. Quickstart Guide

### 1. Launch TRACE (Single Command)
```powershell
python -m trace.main --port 8000
```
Open **`http://localhost:8000`** in any web browser to access the **TRACE Enterprise Operations Console**.

### 2. Run the Full Automated Test Suite
```powershell
python -m pytest -v tests/
```
All 13 unit and invariant tests execute in under 1 second.

### 3. Run the 720-Trial Evaluation Benchmark
```powershell
python -m trace.eval.run_eval
```
Generates statistical metrics with Wilson 95% intervals and writes the complete report to `report/evaluation_report.md`.

---

## 5. Live Demo & Round 2 Twist Playbook

The built-in **Scenario Panel** enables live simulation of any twist requested by the judging panel:

| Panel Twist | Scenario Action | Expected System Reaction |
| :--- | :--- | :--- |
| **Cover the Camera Lens** | Click `1. Cover Lens` | Laplacian sharpness and texture collapse. `CAMERA_OBSTRUCTED` fires in ~2s. Assets decay to `UNKNOWN`. Never calls theft. |
| **Dim Venue Lights** | Click `2. Dim Lights` | Illumination drops to 20%. CLAHE normalizes L channel. Assets remain stable in `SAFE`. |
| **Stand in Front of Asset** | Click `3. Occlude Safe` | Human mask detected ($v < 0.85$). Asset enters `OCCLUDED`. Decay slows ($0.25\times$). `MISSING` is suppressed. |
| **Cut Branch Network** | Click `4. Cut Network` | Sync stops. Signed events accumulate in SQLite `outbox`. Reconnects with zero events lost and `delayed` badge. |
| **Displace Cash Safe** | Click `5. Move Printer` | Drift exceeds threshold for 5 verifications. `MOVED` alert triggers with before/after crops. |
| **Remove Asset** | Click `6. Remove Laptop` | Region visible and matches empty baseline for $T_2$. `MISSING` confirmed on positive evidence. |
| **Leave Foreign Parcel** | Click `7. Unknown Object` | Non-person blob persists for 8s in zone. Triggers `NEW_OBJECT` with crop. |
| **Swap Lookalike Asset** | Click `8. Swap Lookalike` | Visual DNA similarity drops below present threshold / tag mismatch. Triggers `SWAPPED`. |
| **Knock Camera Mount** | Click `9. Knock Camera` | Landmark homography shift > 25px. Triggers `TAMPERED` and prompts re-alignment. |
| **Alter Stored Log** | Click `10. Tamper Ledger` | Simulates insider modifying SQLite row. Click **Verify Ledger** -> turns **RED** identifying exact bad entry! |

---

## 6. API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | TRACE Enterprise Operations Console |
| `GET` | `/api/video/feed` | MJPEG live streaming feed with real-time HUD overlays |
| `GET` | `/api/state` | Current state, belief, and confidence of all assets |
| `POST` | `/api/assets` | Register an asset (wizard polygon, thresholds, tag) |
| `POST` | `/api/assets/{id}/authorize` | Authorize move for $N$ minutes |
| `POST` | `/api/assets/{id}/baseline/accept` | Accept current view as new baseline |
| `GET` | `/api/events` | Search and filter security events |
| `POST` | `/api/events/{id}/ack` | Acknowledge event (signed into ledger) |
| `POST` | `/api/events/{id}/resolve` | Resolve event (signed into ledger) |
| `GET` | `/api/ledger/verify` | Cryptographically verify Ed25519 signatures & SHA-256 chain |
| `GET` | `/api/health` | Camera health, queue depth, fps, and model version |
| `POST` | `/api/demo/inject` | Trigger live fault injection scenarios |
| `WS` | `/ws/stream` | WebSocket streaming connection |
