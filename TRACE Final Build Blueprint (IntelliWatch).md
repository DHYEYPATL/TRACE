# TRACE Final Build Blueprint (IntelliWatch)

Oct 12, 2026 · @dhyey

## 1. Snapshot

TRACE is a software-only asset-confidence engine for the L&T Finance × Tech Tatva IntelliWatch case study. The v3 deck is the Round 1 submission; this doc is the build guide for the prototype, the demo and the pitch.

| Date | Milestone |
| --- | --- |
| 11 Oct | Registration deadline (confirm your team is registered) |
| 12 Oct, 9 AM | Round 1 PPT submission (3 to 5 slides, PPT only) |
| 14 Oct | Round 1 results announced |
| 16 Oct | Final: 10-minute pitch and live demo, then Q&A (subject to L&T Finance executive availability) |

**The idea in one line.** Other systems ask "did it move?". TRACE asks "how sure are we about every asset, right now?", and only alerts on positive evidence.

**Five design ideas that carry the project**

1. **Positive-evidence rule.** "Missing" is raised only when the asset's region is visible, the camera is healthy and the region matches the empty-background baseline. "Can't see it" is never "it's gone".
2. **Doubt Engine.** Every asset has a live confidence that decays while it is unseen. UNKNOWN is a first-class state, so the system never lies and never stays silent.
3. **Privacy by design.** People are segmented in memory and masked before anything is stored. Only asset crops are kept as evidence.
4. **Evidence ledger.** Each event is signed (Ed25519), hash-chained (SHA-256) and its head hash is anchored outside the edge box, so a rewritten history is detectable.
5. **Policy layer.** Fixed vs. mobile assets, after-hours mode and an authorized-move workflow keep alerts rare enough to be trusted in a real office.

**Decisions locked for this build**

- Software-only route: recorded clips plus a scripted event injector, and a live webcam path through the same pipeline.
- One laptop, one Python process for the prototype; the enterprise stack is presented as the production design.
- We do not claim 99.99%. We claim measured numbers with confidence intervals (section 3).
- A semantic detector is optional. Registered assets are verified by region matching; build the detector last.

**How to use this doc.** Sections 2 and 3 are the judges' view. Sections 4 to 8 are the design. Sections 9 and 10 are the stack and starter code. Section 11 is the test plan. Sections 12 to 14 are the demo, the build plan and the checklist.

## 2. Rules and compliance matrix

The deck plus the prototype cover every stated rule, all five judging criteria and all ten demands; the tables below say where and how a judge can verify each one.

### 2.1 Competition rules

| Rule from the brief | How we comply | Where |
| --- | --- | --- |
| Team of 2 to 4, open to MIT Manipal students | Fill in 2 to 4 real names on slide 1 | Slide 1 |
| Round 1 is PPT only, max 3 to 5 slides | 5-slide .pptx, no PDF substitute, opens in PowerPoint | Submission form |
| Round 1 content: problem, architecture, engineering mechanics, validation plan, scalability | One slide each (slides 1 to 5) | Deck v3 |
| Round 2: 10-minute pitch and demo plus Q&A, working prototype required | Laptop prototype with live webcam mode and a scenario panel | Section 12 |
| Round 2 twist: panel adds a live edge case (camera obstruction, network loss, partial occlusion) | Scripted playbook for each, plus fallback video | Section 12 |
| Mock environments and synthetic assets, no real customer data | Cardboard and desk props only; stored evidence has people masked | Slide 2, section 8 |
| Original solution, plagiarism disqualifies | Own code; keep a CREDITS file listing every library, model and licence used | Section 9 |
| Minimum baseline: register a baseline, detect at least one event, give an evidence trail or confidence indicator, raise a real-time alert with an audit record | All four implemented, plus confidence on every asset | Section 5 to 7 |

### 2.2 Judging criteria to evidence

| Criterion (weight) | What the panel looks for | Our evidence | What to show |
| --- | --- | --- | --- |
| Technical depth and engineering quality (25%) | Detection algorithm efficiency, edge/cloud architecture, code quality | Cheap region matching gated by a pixel check, positive-evidence state machine, edge/server split, tests and typed code | Architecture slide, repo tree, passing unit tests, per-stage latency |
| Prototype, demonstration and validation (25%) | A working simulation that detects changes and logs evidence | Live demo, evaluation report, ablation against a naive baseline | Confusion matrix, latency histogram, confidence intervals |
| Innovation and differentiation (20%) | Originality, privacy-preserving design | Doubt Engine, on-device people masking, off-box anchored ledger | Browse the evidence folder live: no faces anywhere |
| Problem understanding and relevance (15%) | Practicality inside an L&T Finance branch | Safes, gold valuation machines, printers, mobile assets, after-hours policy, per-branch cost | Floor-plan demo with branch-like props, policy screen |
| Feasibility, security and defense (15%) | Handling the live edge case, trade-offs, hardware limits | Threat model, edge-case playbook, honest limits slide, Q&A bank | Section 12 |

### 2.3 The ten demands to module, test and demo moment

| # | Demand | Module | Acceptance test | Demo moment |
| --- | --- | --- | --- | --- |
| 1 | Asset registration | Registration wizard, `assets` table | Register ID, zone, class, policy and 3+ reference photos in under 2 minutes; asset shows SAFE | Opening minute |
| 2 | Baseline creation | Baseline builder | Median of about 30 frames stored with embedding, keypoints, bbox and empty-region crop; "Accept new baseline" works | Opening minute |
| 3 | Asset detection | Region verifier, optional detector | Registered assets matched in clean footage with the match score above threshold in 99%+ of idle frames per asset (target, to be measured) | Live view |
| 4 | Movement detection | Drift module | Move beyond the class threshold alerts within 2 s (P95); jitter below the threshold does not | Push the safe model |
| 5 | Missing detection | Positive-evidence rule | Removal alerts after the configured wait only when the region is visible and empty; a hand held over it for 20 s does not | Remove laptop, then occlude |
| 6 | Unexpected asset | Background model, NEW\_OBJECT | An unregistered box placed in the zone for the configured time raises NEW\_OBJECT with a crop | Place an unknown box |
| 7 | Anomaly classification | State machine | Moved, missing, newly detected, camera obstruction and system offline all exist as event types, plus swapped, state-changed, tampered, unknown | Throughout |
| 8 | Evidence generation | Evidence writer, ledger | Each event has before and after crops, timestamp, asset ID, change type, confidence and history; Verify ledger turns green, a hand-edited row turns it red | Ledger tamper test |
| 9 | Alerting and dashboard | Alert router, UI | Current status per asset, Telegram alert in under 3 s, searchable incident log | Phone buzzes live |
| 10 | Robustness and validation | Evaluation harness | Report under lighting change, partial occlusion, obstruction, network failure and false-positive scenarios, with accuracy and response time | Slide 4, report |

## 3. The accuracy question (99.99%)

Do not claim 99.99%: it cannot be demonstrated in a four-day project and a panel can check the arithmetic. Claim measured numbers with confidence intervals, and design for no silent failures and near-zero false alerts.

### 3.1 Why 99.99% is not provable here

With zero errors in n independent trials, the 95% upper bound on the error rate is about 3/n (the "rule of three"). Showing an error rate of 0.01% or lower needs about 30,000 error-free trials.

```latex
n \ge \frac{3}{p_{max}} = \frac{3}{10^{-4}} = 30{,}000
```

- **Our test size.** 720 events with zero errors gives a Wilson 95% lower bound of about 99.5% (99.47%). That is the honest ceiling for the claim.
- **Frames are not trials.** Consecutive frames are highly correlated, so 100,000 clean frames is not 100,000 independent trials.
- **Rates need camera-days.** Zero false alarms in T camera-days only bounds the rate at 3/T per day. One idle day proves under 3 per day; proving under 1 per day needs at least 3 clean camera-days. Start the idle run by the evening of 13 Oct (for example 2 cameras for 36 hours each) so it finishes before 16 Oct.

### 3.2 What we claim instead

| Metric | Target | How it is measured |
| --- | --- | --- |
| No silent failure | 100% by design: camera blind, offline or occluded never shows SAFE | Fault-injection run: 30 injected faults, every one surfaced as UNKNOWN, OBSTRUCTED or OFFLINE |
| Precision of auto-confirmed alerts | 99% or higher | True alerts over all alerts in the evaluation set, with a Wilson 95% interval |
| Recall on MOVED, MISSING, SWAPPED | 97% or higher | Scripted events matched to alerts (rule in section 11) |
| False alarms | Under 1 per camera per day | Idle footage, at least 3 camera-days |
| Latency | MOVED P95 under 2 s; CAMERA\_OBSTRUCTED P95 under 3 s; MISSING = configured wait plus under 2 s | Timestamps from event injection to alert delivery |
| Ledger tamper detection | 100% | At least 30 edit, delete and full-chain-rewrite attempts |

These are targets until measured. If one misses, show the measured number and the fix; an honest 96% beats an unprovable 99.99%.

### 3.3 How the design pushes error toward zero

1. **Constrain the problem.** Fixed camera, registered assets and region-only processing remove most ways to be wrong.
2. **Temporal voting.** A decision needs N consecutive frames or T seconds. Independent frame noise vanishes geometrically; correlated causes (lighting, camera shake) are handled by the health checks and multi-lighting baselines.
3. **Agreement of evidence.** MOVED or SWAPPED needs the pixel change, the embedding score and the keypoint inlier ratio to agree.
4. **Hysteresis.** Separate enter and exit thresholds stop flapping at the boundary.
5. **Selective prediction.** Confidence 0.9 or higher auto-alerts; 0.6 to 0.9 goes to a "Needs review" queue with no push; below 0.6 is logged only. Calibrate with temperature scaling and show the reliability diagram.
6. **Positive-evidence rule.** Removes the biggest false-alarm class: occlusion and camera trouble read as theft.
7. **Policy layer.** Removes the second biggest: legitimate moves of mobile assets.
8. **Feedback loop.** "False positive" saves a hard negative and nudges that asset's threshold within fixed bounds.
9. **Redundancy (roadmap).** Two cameras for high-value assets and an optional tilt or reed sensor on safes.

### 3.4 What to say on stage

"We designed for no silent failures and near-zero false alerts, and here is what we measured, with intervals. A claim of 99.99% would need roughly 30,000 clean trials; we would rather show you numbers you can check."

## 4. Solution concept and architecture

TRACE splits into an edge pipeline that sees and decides, and a server that stores, alerts and shows. The prototype runs both on one laptop; every production part is a drop-in replacement for its prototype twin.

### 4.1 Prototype vs production

| Layer | Prototype (we build) | Production (we present) |
| --- | --- | --- |
| Input | Recorded clips plus laptop webcam | RTSP IP cameras (H.264) |
| Edge compute | Laptop, one Python process | Raspberry Pi 5, Intel N100 mini-PC or Jetson Orin Nano |
| Edge storage | SQLite (WAL mode) plus a crops folder | Same, plus encrypted ring buffer and disk quotas |
| Transport | In-process queue, HTTP to FastAPI | MQTT (Mosquitto) and HTTPS with store-and-forward |
| Server | FastAPI with SQLite | FastAPI, PostgreSQL, MinIO (object lock), Redis |
| Dashboard | Streamlit | React with WebSocket (or keep Streamlit) |
| Alerts | Telegram bot, on-screen banner | Telegram, email, SMS gateway, webhook to the ticketing tool |
| Identity | One local admin | Keycloak SSO with roles |
| Observability | Logs and a `/metrics` endpoint | Prometheus, Grafana, Loki |
| Updates | `git pull` | Signed OTA bundles for models and config |

### 4.2 Components and what each owns

- **Capture:** reads frames from a clip, webcam or RTSP URL on its own thread; drops old frames instead of queueing.
- **Health monitor:** blur, brightness, edge density, frozen frame, landmark visibility. Decides CAMERA\_OBSTRUCTED.
- **Aligner:** keeps the view registered to the reference frame; separates "camera moved" from "asset moved".
- **People segmenter:** produces the person mask used for occlusion reasoning and for masking evidence.
- **Change gate:** cheap per-region pixel check that decides whether the expensive matcher runs.
- **Region verifier:** embedding plus keypoint match of each registered asset against its baseline.
- **Relocator (optional):** finds a moved asset elsewhere in the frame; the only place a semantic detector is used.
- **Belief engine:** per-asset state machine with decaying confidence (section 6).
- **Policy engine:** decides whether a state change becomes an alert, how severe, and to whom.
- **Evidence writer:** masked before and after crops, signed hash-chained ledger entry.
- **Outbox and sync:** persists events locally and delivers them to the server idempotently.
- **Alert router and dashboard:** fan-out to Telegram and the UI; ack, resolve, false-positive and authorized-move actions.
- **Scenario injector (demo only):** buttons that cover the lens, cut the network, occlude an asset or play a scripted move.

### 4.3 Architecture diagram

&#91;embedded content: architecture · edge pipeline, server and dashboard\]

Read it inside each box top-down: the edge runs the whole pipeline and signs every event, and the server only stores, alerts and shows, so losing the server never stops detection.

### 4.4 Per-frame data flow and budgets

These are budgets to benchmark on your laptop, not measurements.

1. **Capture** at about 15 fps; **process** every third frame (5 fps), plus immediately when the change gate trips.
2. **Health check** on every processed frame, under 5 ms.
3. **Align** to the reference every second or when drift is suspected, under 30 ms.
4. **Segment people** on a downscaled frame at 5 fps, under 60 ms; dilate the mask.
5. **Change gate** per asset region (CLAHE then SSIM), under 3 ms per asset.
6. **Verify** when the gate trips, and at least once per second per asset: embedding plus keypoint match; cache embeddings of unchanged regions.
7. **Update belief**, apply policy, emit an event if a transition qualifies.
8. **Persist and deliver:** mask crops, append to the ledger, queue to the server, alert router, dashboard. Target end-to-end for MOVED: P95 under 2 s.

**Prototype process model.** A capture thread feeds the pipeline loop; uvicorn serves the API in a thread of the same process; Streamlit runs as a second process and only talks to the API; both share SQLite in WAL mode.

## 5. Core algorithms and parameters

Everything below is classical computer vision plus one small pretrained embedding model, so nothing needs GPU training and every threshold is a named parameter you can tune. All numbers are starting values to calibrate on your own footage (1280×720 assumed).

### 5.1 Registration and baseline (demands 1 and 2)

1. Pick the camera and capture a **reference frame**; this anchors alignment (5.3).
2. Draw the asset's **region** (polygon) and its **expected zone** (default: region grown by 10%).
3. Enter ID, name, class (safe, gold machine, printer, laptop, stationery, camera, chair, other), criticality (1 to 3), policy (fixed or mobile) and an optional tag ID.
4. Capture **3 or more reference photos in at least 2 lighting conditions**. For each condition record about 30 frames and keep the 8 sharpest; the baseline image is their per-pixel median.
5. **Visual DNA:** DINOv2-small embedding (384-dimensional, L2-normalised) of each photo, plus augmented variants (brightness ±30%, contrast, light blur, shifts of ±3%). The spread of similarities between variants gives the per-asset noise level.
6. **Keypoints:** SIFT descriptors inside the region of the median image. Count them: under 40 means a low-texture asset (plain cardboard), which switches matching to embedding-only mode (5.6).
7. **Empty-region capture (recommended):** ask the operator to remove the asset and record 30 frames; the median is `empty_roi`. It is what makes MISSING provable. If skipped, MISSING runs in a reduced-evidence mode and says so.
8. **Calibrate:** set the asset's present-threshold from the variant similarities (mean minus 3 standard deviations, clipped to 0.55 to 0.90).
9. **Dry run:** the wizard shows 10 s of stable SAFE, then asks for a nudge to confirm MOVED fires. Only then does it activate.

**Demo tip.** Stick a small unique ArUco or QR tag on every mock asset. Plain cardboard has almost no texture; a tag makes matching and identity near-trivial and also separates identical look-alikes.

### 5.2 Camera health: obstruction and offline (demand 7)

Compute per processed frame, each compared with a rolling baseline of the last good 60 s:

- **Sharpness:** variance of the Laplacian; fail below 0.35× baseline.
- **Brightness:** mean of V in HSV; fail below 0.15 (0 to 1 scale) or a drop over 60% within 2 s.
- **Texture:** Canny edge fraction per tile of a 4×4 grid; fail when more than 60% of tiles fall below 0.3× baseline.
- **Landmark visibility:** share of reference keypoints or ArUco tags still matched; fail below 40%.

**CAMERA\_OBSTRUCTED** when at least 2 of the 4 fail for 1.5 s; clears when all pass for 3 s. A hand over the lens collapses sharpness and texture together, so it fires in about 2 s. **SYSTEM\_OFFLINE (camera)** when frames stop for 3 s, reads error out, or the frame is bit-identical for 2 s (frozen feed). **SYSTEM\_OFFLINE (edge)** is raised by the server when heartbeats stop for 15 s.

### 5.3 Alignment and camera-bump detection

- Place 4 to 6 **ArUco markers** (dictionary 4×4\_50, OpenCV) on static surfaces, or use natural ORB features of the fixed background.
- Every second, estimate a homography between current and reference landmarks (RANSAC).
- Mean landmark shift under 3 px: ignore. From 3 to 25 px: warp regions by the homography, log "camera drift". Over 25 px or under 30% inliers: **TAMPERED (camera moved)**, suspend MOVED decisions for that camera and ask the operator to re-anchor.

### 5.4 People mask and occlusion (privacy and demand 5)

- Segment people with MediaPipe's image segmenter (Apache-2.0) at 256×256 (or YOLO11n-seg, AGPL, fine for the competition). Upscale, dilate 15 px, OR over the last 3 frames.
- **Occlusion ratio** of a region: share of its pixels under the person mask. Above 0.15, the region is a candidate for OCCLUDED.
- **Visibility** v = share of region pixels covered neither by the mask nor by a transient moving blob. MISSING and SWAPPED are never decided with v below 0.85.
- **Masking evidence:** blur person pixels (Gaussian kernel 51) in every saved crop and set `person_masked = true`. No face recognition, no face storage, no raw video at rest.

### 5.5 Change gate

For each aligned region, apply CLAHE (clip limit 2.0, 8×8 tiles) to the L channel of LAB on both live and baseline crops, then compute SSIM and mean ΔE on non-person pixels. If SSIM is 0.93 or higher and ΔE is 6 or lower for the last N frames the region is "unchanged" and the expensive matcher is skipped, except for a once-per-second identity check.

### 5.6 Region verification (demand 3)

- **Embedding score** s\_emb: the mean of the top 3 cosine similarities between the live crop's embedding and the stored variants.
- **Keypoint score** s\_kp: RANSAC inlier count of SIFT matches (Lowe ratio 0.75) against the median baseline; need 12 or more inliers.
- **Fusion:** s = 0.6·s\_emb + 0.4·min(1, inliers/40). In embedding-only mode (low texture) s = s\_emb and a tag, if present, must also decode to the right ID.
- An asset is **present** when s is at or above its present-threshold for 3 consecutive verifications.

### 5.7 Decision rules

| Event | Rule (all conditions must hold) |
| --- | --- |
| MOVED (demand 4) | Region is matched but drift d = distance of centroid from baseline over baseline diagonal exceeds the class threshold for 5 consecutive verifications (at least 1 s). Or the relocator finds the asset outside its zone. |
| MISSING (demand 5) | Visibility 0.85 or higher, camera healthy, region matches `empty_roi` (SSIM 0.85 or higher), relocator finds nothing, for the configured wait T2. |
| SWAPPED | Region is occupied (SSIM to `empty_roi` below 0.7) and visible, but s is below the present-threshold for 10 s; or the tag ID mismatches (immediate). |
| NEW\_OBJECT (demand 6) | Foreground blob of area 0.5% of the frame or more, inside a monitored zone, not under the person mask, not on a registered region, persistent for T\_new, and matching no registered asset. If it matches a registered asset elsewhere, it is MOVED, not new. |
| STATE\_CHANGED | State classifier (5.8) at 0.85 or higher for 5 votes. |

The background model for NEW\_OBJECT is a running median of the static scene, updated only when no person is present and nothing has changed for 30 s.

### 5.8 State change classifier

Few-shot, no deep training: collect 15 to 20 mock images per state (safe door open or closed, printer cover on or off, cable plugged or unplugged), embed the asset's sub-region with DINOv2-small, and fit logistic regression or kNN (k=5). Retraining takes seconds, so you can add a state during the demo prep. The safe door open or closed is the showpiece.

### 5.9 Calibration and thresholds

Calibrate confidence by temperature scaling on held-out data and show the reliability diagram. Decision bands: 0.9 or higher alerts, 0.6 to 0.9 goes to the review queue, below 0.6 is logged. **Split your recordings: about 40% for tuning, 60% for the reported test.** Never tune on the test set.

### 5.10 Starting parameter table

| Parameter | Start value | Tune with |
| --- | --- | --- |
| Process rate | 5 fps | CPU load |
| MOVED threshold (fraction of baseline diagonal) | safe 0.10, gold machine 0.10, printer 0.15, laptop 0.25, stationery 0.30, chair 0.35 | Move trials |
| MOVED votes | 5 consecutive verifications | Latency vs. false alarms |
| MISSING wait T2 | Demo 15 s, production 60 s or more | Policy |
| NEW\_OBJECT persistence T\_new | Demo 8 s, production 20 s | Idle footage |
| SWAPPED persistence | 10 s | Lighting sweeps |
| Visibility floor | 0.85 | Occlusion trials |
| Obstruction hold and release | 1.5 s and 3 s | Lens-cover trials |
| Unchanged gate | SSIM 0.93, ΔE 6 | Idle footage |
| Present-threshold | 0.55 to 0.90, per asset | Registration variants |

## 6. State machine, Doubt Engine and policy

Every asset runs the same state machine, fed by the decision rules in section 5; a transition becomes an alert only if the policy engine says so.

&#91;embedded content: state machine · per-asset states, with the camera override\]

A state is entered only after its votes or timer; every alert state returns through RETURNED, and the override band applies to every box above it.

### 6.1 States

| State | Meaning | Severity | Alerts? |
| --- | --- | --- | --- |
| SAFE | Matches baseline, inside its zone | none | No |
| OCCLUDED | Not visible: a person or object covers the region | none | No |
| UNKNOWN | Belief below 0.5: camera blind, offline, or asset unseen too long | low | Batched "unverified" notice after 2 min |
| MOVED | Displaced beyond the class threshold | medium (high for critical assets) | Yes |
| SWAPPED | Region occupied by a different object, or tag mismatch | high | Yes |
| STATE\_CHANGED | State classifier flipped (for example safe door opened) | by policy | Yes |
| MISSING | Region proven empty and asset not found elsewhere | high | Yes |
| NEW\_OBJECT | Unregistered object persists in a monitored zone | medium | Yes |
| TAMPERED | Camera moved, or vision and sensor disagree | high | Yes |
| RETURNED | Back in its zone after an incident | none | Closes the incident |

Per-camera system states: **CAMERA\_OBSTRUCTED** (high) and **SYSTEM\_OFFLINE** (high).

### 6.2 Transitions and timers

| From to | Guard | Hysteresis |
| --- | --- | --- |
| SAFE to OCCLUDED | Visibility below 0.85 | Exit when visibility is 0.9 or higher for 2 s |
| SAFE to MOVED | Drift above threshold for 5 verifications, visibility 0.85 or higher | Exit to RETURNED when back within half the threshold for 5 verifications |
| SAFE to SWAPPED | Occupied and score below present-threshold for 10 s, or tag mismatch | Exit when score is at or above threshold for 5 s |
| SAFE to STATE\_CHANGED | Classifier 0.85 or higher for 5 votes | Reverse flip for 5 votes |
| SAFE or OCCLUDED to MISSING | Positive evidence of emptiness for T2 | Exit to RETURNED when the asset matches again |
| Any to UNKNOWN | Belief C below 0.5 | Exit on a verified sighting (C resets to 1) |
| Any to CAMERA\_OBSTRUCTED | Health check fails (5.2) | Assets freeze to OCCLUDED or UNKNOWN; MISSING is suppressed |
| RETURNED to SAFE | Stable for 5 s | None |

**Invariants that never break**

1. MISSING and SWAPPED need visibility 0.85 or higher and a healthy camera.
2. While a camera is CAMERA\_OBSTRUCTED or SYSTEM\_OFFLINE, none of its assets can show SAFE; they decay to UNKNOWN.
3. Every asset always shows a state, a belief value and a last-verified time.
4. No alert fires on a single frame; every alert state needs its vote count or timer.
5. Nothing is ever deleted: resolving an incident appends an event.

### 6.3 The Doubt Engine

Belief C is a number from 0 to 1 per asset, shown as a meter on the dashboard. It is separate from an event's confidence, which is the calibrated probability that an alert is real and is stored with its evidence.

```latex
C(t) = C_0 \, e^{-\lambda_c \, (t - t_{seen})}, \qquad \lambda_c = \frac{\ln 2}{T_{half,c}}
```

- A **verified sighting** (score at or above threshold, healthy camera, visibility 0.85 or higher) resets C to 1.
- C reaches 0.5 after one half-life, which is the UNKNOWN threshold.
- While **OCCLUDED** the decay rate is multiplied by 0.25 (not seeing it is expected); when blind or offline it runs at full rate.

| Asset class | Half-life, production | Half-life, demo |
| --- | --- | --- |
| Safe, gold machine | 30 min | 40 s |
| Printer | 15 min | 30 s |
| Laptop, chair | 5 min | 20 s |
| Stationery | 2 min | 15 s |

### 6.4 Policy engine

| Asset mode and time | Alerts |
| --- | --- |
| Fixed, any time | MOVED, SWAPPED, MISSING, STATE\_CHANGED |
| Mobile, business hours | MOVED logged only; MISSING only after 30 min gone |
| Mobile, after hours | MOVED and MISSING alert |
| Criticality 3 (safe, gold machine), after hours | Any door or state change is critical and escalates |

- **Authorized move.** Staff press "Authorize move, 10 min" on the dashboard or send `/authorize printer-01 10` to the bot. MOVED and MISSING are suppressed for that asset for the window, the actor is logged, and afterwards the system expects RETURNED or asks "Accept new location?".
- **Dedupe and cooldown.** One alert per asset and event type; repeats within 5 min fold into the same incident counter.
- **Escalation.** Unacknowledged high-severity alerts re-notify after 5 min and escalate to a second contact after 10 min; medium re-notifies after 15 min.
- **Routing by severity.** Low: dashboard. Medium: dashboard and Telegram. High: all of that, plus a local banner or siren.
- **Fatigue guards.** UNKNOWN is batched into one message per camera; at most N pushes per hour per camera; configurable quiet hours.

## 7. Data model, API, events and config

Thirteen tables, one event shape and about a dozen endpoints are enough for the whole prototype; the schema is the same in SQLite and PostgreSQL.

### 7.1 Tables

| Table | Key columns | Notes |
| --- | --- | --- |
| `cameras` | id, name, source\_uri, status, last\_heartbeat, reference\_frame\_path, landmarks\_json | One row per feed |
| `assets` | id, name, class, criticality, policy\_mode, camera\_id, region\_polygon, zone\_polygon, tag\_id, present\_threshold, active | Registration output |
| `baselines` | id, asset\_id, kind (median, empty, variant), image\_path, embedding\_blob, keypoints\_blob, lighting\_label | Several rows per asset |
| `states` | asset\_id, state, belief, confidence, last\_verified\_at, updated\_at | Current state, one row per asset |
| `events` | id (UUIDv7), asset\_id, camera\_id, type, severity, confidence, status, detected\_at, received\_at, before\_path, after\_path, payload\_json, incident\_id | `received_at` minus `detected_at` drives the "delayed" badge |
| `incidents` | id, asset\_id, opened\_at, closed\_at, status, counter | Folds repeats |
| `ledger` | idx, body, hash, sig | Append-only; see section 10 |
| `anchors` | id, ledger\_idx, head\_hash, channel, anchored\_at, ref | Off-box proof of the head |
| `actions` | id, event\_id, actor, action (ack, resolve, false\_positive, authorize, escalate), note, at | Human actions |
| `authorizations` | asset\_id, actor, starts\_at, ends\_at | Authorized-move windows |
| `policies` | id, asset\_id, yaml | Versioned |
| `heartbeats` | source, at, queue\_depth, fps, model\_version | Drives SYSTEM\_OFFLINE |
| `users` | id, name, role | admin, operator, viewer |

Protect `ledger` with a trigger that rejects UPDATE and DELETE; in production add database roles that cannot alter it.

### 7.2 Event shape

```json
{
  "id": "0192b7c4-3f1a-7c2e-9d4b-2a6f8e1c5d70",
  "asset_id": "safe-02",
  "camera_id": "cam-1",
  "type": "MOVED",
  "severity": "high",
  "confidence": 0.93,
  "detected_at": "2026-10-16T11:42:07.381+05:30",
  "evidence": {
    "before": "evidence/2026-10-16/0192b7c4/before.jpg",
    "after": "evidence/2026-10-16/0192b7c4/after.jpg",
    "person_masked": true,
    "sha256_before": "ab12…",
    "sha256_after": "c9f0…"
  },
  "details": {"drift_frac": 0.21, "threshold": 0.10, "votes": 5, "visibility": 0.97, "camera_health": "ok"},
  "ledger": {"idx": 412, "hash": "7d3e…"}
}
```

### 7.3 REST API

| Method and path | Purpose |
| --- | --- |
| `POST /api/assets` | Register an asset (wizard output) |
| `GET /api/assets`, `GET /api/assets/{id}` | List and read |
| `POST /api/assets/{id}/baseline` | Capture median, variant or empty baselines |
| `POST /api/assets/{id}/baseline/accept` | Accept the current view as the new baseline |
| `POST /api/assets/{id}/authorize` | Authorize a move for N minutes |
| `GET /api/state` | Current state and belief of every asset |
| `GET /api/events` | Search: asset, type, status, from, to, free text |
| `GET /api/events/{id}` | One event with evidence URLs and history |
| `POST /api/events/{id}/ack`, `/resolve`, `/false-positive` | Human actions |
| `GET /api/ledger/verify` | Returns ok, entries checked and first bad index |
| `GET /api/ledger/head` | Current head hash and last anchor |
| `GET /api/health`, `GET /metrics` | Cameras, queue depth, fps, latency, model version |
| `POST /api/ingest/events` | Edge to server, idempotent on event id (split mode) |
| `POST /api/demo/inject` | Demo only: cover lens, cut network, occlude, scripted move |
| `WS /ws/stream` | Pushes `state_update`, `event_new`, `event_update`, `health` |

In production the edge also publishes MQTT with QoS 1: `trace/{branch}/{camera}/heartbeat` (retained, with a last-will message that becomes SYSTEM\_OFFLINE), `.../event` and `.../state`.

### 7.4 Config file

```yaml
site: demo-branch-01
timezone: Asia/Kolkata
profile: demo                  # demo or production: picks half-lives and timers
business_hours: {days: [Mon, Tue, Wed, Thu, Fri, Sat], start: "09:30", end: "18:30"}
cameras:
  - id: cam-1
    source: webcam:0           # or file:clips/idle.mp4, rtsp://...
    process_fps: 5
assets:
  - {id: safe-02,   class: safe,   criticality: 3, mode: fixed,  camera: cam-1, tag_id: 12, thresholds: {move_frac: 0.10, present: 0.72}}
  - {id: laptop-03, class: laptop, criticality: 1, mode: mobile, camera: cam-1}
timers:
  demo:       {missing_wait_s: 15, new_object_s: 8,  swap_s: 10}
  production: {missing_wait_s: 60, new_object_s: 20, swap_s: 10}
alerts:
  telegram: {enabled: true, chat_id: "${TELEGRAM_CHAT_ID}"}
  escalate_after_s: {high: 300, medium: 900}
ledger:
  anchor_every_s: 60
  anchor_channels: [telegram, local_file]
privacy: {mask_people: true, retention_days: 30}
```

## 8. UI and workflow

Seven screens and one demo-only panel, built so an operator sees every asset's status at a glance and resolves an incident in two clicks.

### 8.1 Visual rules

- Dark theme with the deck's palette, so the pitch and the product look like one thing.
- State colours stay constant everywhere: SAFE green, MOVED rose, UNKNOWN amber, CAMERA\_OBSTRUCTED cyan, OCCLUDED grey, SWAPPED and MISSING a stronger rose with an icon. Never colour alone: every chip also carries a label.
- Confidence is always visible next to a status; evidence is one click away.
- Status first, detail on demand: the dashboard shows no raw numbers beyond belief and latency.

### 8.2 Screens

| Screen | Purpose | Key elements | Primary actions |
| --- | --- | --- | --- |
| 1. Dashboard | Current truth at a glance | Camera view with region overlays, asset list with state and belief meter, live alert feed, KPI strip (assets verified, cameras online, P95 latency) | Open incident, Ack |
| 2. Register wizard | Add an asset in under 2 min | 5 steps (8.4) | Activate |
| 3. Live view | Debug and demo | Feed with region overlays, state labels, person mask toggle, health badges | Toggle overlays |
| 4. Incident detail | Evidence and decision | Before and after slider, confidence ring, timeline, signature and hash status | Ack, Resolve, False positive, Authorize move, Escalate, Accept new baseline |
| 5. Incident log | Audit and search | Filters (asset, type, severity, status, date), free-text search, CSV export, **Verify ledger** button (green or red) | Export, Verify |
| 6. System health | Trust in the system | Camera heartbeats, queue depth, fps, model version, latency histogram | None |
| 7. Settings and policy | Tune without code | Per-class thresholds and half-lives, business hours, notification routing, authorized-move defaults | Save |
| Demo panel | Round 2 only | Scenario buttons (8.6) | Inject |

### 8.3 Wireframes

```text
DASHBOARD
+--------------------------------------------------------------------------+
| TRACE  demo-branch-01     Assets 6/6 verified   Cams 1/1 online   P95 1.4s |
+----------------------------------+---------------------------------------+
| CAMERA / FLOOR PLAN              | ASSETS                                |
|  [safe-02]    [printer-01]       | safe-02     SAFE      belief 1.00  ### |
|        [laptop-03]   [chair-04]  | printer-01  MOVED     belief 0.93  ##- |
|  region overlays in state colour | laptop-03   UNKNOWN   belief 0.41  #-- |
|                                  | chair-04    OCCLUDED  belief 0.88  ##- |
+----------------------------------+---------------------------------------+
| LIVE ALERTS                                                              |
| 11:42:07  HIGH  printer-01 MOVED    conf 0.93   [Ack] [Open]             |
| 11:41:50  LOW   laptop-03 UNKNOWN   camera partly blocked   [Open]       |
+--------------------------------------------------------------------------+

INCIDENT DETAIL
+--------------------------------------------------------------------------+
| printer-01  MOVED  HIGH   conf 0.93   11:42:07   status: open            |
+--------------------------------+-----------------------------------------+
| BEFORE  <----slider---->  AFTER | Timeline                                |
| (people masked)                 | 11:42:05  drift detected (vote 1/5)     |
|                                 | 11:42:07  alert raised                  |
| Drift 21% of size (limit 10%)   | 11:42:31  acknowledged by operator      |
| Visibility 0.97  Camera ok      | Ledger #412  signature OK  anchored     |
+--------------------------------+-----------------------------------------+
| [Ack] [Resolve] [False positive] [Authorize move] [Escalate] [Re-baseline]|
+--------------------------------------------------------------------------+
```

### 8.4 Register wizard (screen 2)

1. **Camera.** Pick the feed; the wizard captures the reference frame and finds landmarks.
2. **Region.** Draw the asset polygon and its expected zone on the frozen frame.
3. **Details.** ID, name, class, criticality, mode (fixed or mobile), optional tag.
4. **Baselines.** Capture 3 or more photos in 2 lighting conditions, then the empty-region capture. Each step shows a progress bar and a sharpness check.
5. **Dry run.** 10 s of stable SAFE, then a guided nudge to prove MOVED fires. Then Activate.

### 8.5 Operator workflows

**Alert handling.** The phone buzzes with asset, event, confidence and an "Open" link. The operator opens the incident, compares before and after, then picks one of four outcomes: acknowledge (expected and fine), authorize move (it was staff), false positive (teaches the system), or escalate. Every outcome appends an action row and a ledger entry.

**Review queue.** Events with confidence 0.6 to 0.9 never push; they wait in a "Needs review" list on the dashboard. Confirming one promotes it to an alert; dismissing one saves a hard negative.

**After a legitimate move.** Authorize the move, move the asset, then accept the new location so the baseline updates and the incident closes.

### 8.6 Demo scenario panel (virtual faults)

The buttons apply faults to the frame stream **before** the pipeline, so the exact production code path reacts. With physical props the same events happen for real; the panel is the safety net.

| Button | What it does to the stream |
| --- | --- |
| Cover lens | Blurs and darkens the whole frame for 5 s |
| Dim lights | Applies a gamma and gain change |
| Occlude asset | Draws a person-shaped mask over one region |
| Cut network | Stops edge-to-server sync; alerts continue locally |
| Move asset | Plays a scripted clip or warps the region |
| Remove asset | Replaces the region with its empty baseline |
| Place unknown object | Pastes an unregistered object into the zone |
| Swap asset | Pastes a look-alike into the region |
| Knock camera | Applies a small homography shift |
| Tamper ledger | Edits one stored row, then Verify ledger shows red |

### 8.7 Build notes for Streamlit

- Refresh status with `st.fragment` and `run_every` (Streamlit 1.37 or later) instead of rerunning the whole page.
- For smooth live video, serve an MJPEG stream from FastAPI and embed it as an image; `st.image` refreshes flicker.
- Draw regions with `streamlit-drawable-canvas` (polygon tool), or run OpenCV `selectROI` as a one-off desktop tool.
- Send Telegram alerts with plain `requests.post` to the Bot API; no heavy library needed.
- Set the dark theme in `.streamlit/config.toml` using the deck's colours.
- If Streamlit feels laggy under load, FastAPI plus htmx and Server-Sent Events is the lightweight fallback.

## 9. Tech stack, licences and hardware

Every component is free and open source or a free service; the prototype needs only a laptop with 8 GB RAM or more, no GPU. Verify each licence yourself before any commercial use, and keep a CREDITS file in the repo.

### 9.1 Stack

| Layer | Tool | Licence | Used for | Fallback |
| --- | --- | --- | --- | --- |
| Vision | OpenCV (ArUco, SIFT, CLAHE, SSIM helpers) | Apache-2.0 | All classical CV | None needed |
| Inference | ONNX Runtime | MIT | CPU inference of exported models | PyTorch CPU |
| Embeddings | DINOv2-small (Meta) | Apache-2.0 (confirm on the model card) | Visual DNA, state classifier | MobileNetV3 features |
| People mask | MediaPipe Image Segmenter | Apache-2.0 | Occlusion and privacy masking | YOLO11n-seg (AGPL-3.0) |
| Detector (optional) | OWLv2 via Hugging Face Transformers, or fine-tuned YOLO11n | Apache-2.0 / AGPL-3.0 | Re-finding moved assets | Production: YOLOX or RT-DETR (Apache-2.0) |
| Tracking (optional) | ByteTrack | MIT | Blob continuity | Nearest-centroid |
| Statistics | NumPy, SciPy, scikit-learn, statsmodels | BSD | Metrics, calibration, Wilson intervals | None needed |
| Backend | FastAPI, Uvicorn, Pydantic | MIT, BSD, MIT | API, WebSocket, validation | Flask |
| Database | SQLite, then PostgreSQL | Public domain, PostgreSQL licence | Everything | None needed |
| Object store (production) | MinIO | AGPL-3.0 | Evidence with object lock | SeaweedFS (Apache-2.0) |
| Dashboard | Streamlit, streamlit-drawable-canvas | Apache-2.0, MIT | Prototype UI | React and Vite |
| Crypto | PyNaCl | Apache-2.0 | Ed25519 signing | `cryptography` |
| Time anchor | freetsa.org (free RFC 3161 service), OpenTimestamps | Free service, LGPL-3.0 client | Off-box proof of the ledger head | Telegram message plus local file |
| Messaging | Telegram Bot API, Mosquitto (production) | Free service, EPL-2.0 | Alerts, edge to server | ntfy for push |
| Observability | Prometheus, Grafana, Loki | Apache-2.0, AGPL-3.0, AGPL-3.0 | Production monitoring | Plain logs |
| Identity | Keycloak | Apache-2.0 | Production SSO and roles | Local users |
| Labelling | CVAT, Label Studio | MIT, Apache-2.0 | Ground truth, state images | Spreadsheet |
| Dev tooling | uv, pytest, ruff, mypy, pre-commit, Docker Compose | MIT, Apache-2.0 | Reproducible, tested, typed code | pip and venv |

AGPL-3.0 tools (YOLO11, MinIO, Grafana, Loki) are fine for a competition and for self-hosting; their network-use obligations matter if you modify them and offer them as a service, which is why the production plan prefers Apache-2.0 alternatives where one exists.

### 9.2 Offline readiness

Campus Wi-Fi may block Hugging Face, Telegram or both on 16 Oct.

- Download every model weight in advance and copy the Hugging Face cache to the demo laptop; test with Wi-Fi off.
- Pin versions in `uv.lock` and keep a wheel cache.
- Add ntfy as an alert fallback and test it on a mobile hotspot.
- Carry the pre-recorded fallback video (section 12).

### 9.3 Hardware and props

| Item | Purpose |
| --- | --- |
| Laptop, 8 GB RAM or more, 4 or more cores | Runs everything; GPU not required |
| 1080p USB webcam, or a phone as a webcam (DroidCam, Iriun or OBS virtual camera) | Live feed |
| Tripod or fixed mount | Camera must not move between runs |
| Mobile hotspot or small router | Network-loss demo and Wi-Fi fallback |
| Second phone | Receives Telegram alerts on stage |
| Props: cardboard safe with a hinged door, printer-sized box, laptop, stationery set, camera model, chair | Mock assets |
| Printed ArUco stickers | Asset tags and wall landmarks |
| Desk lamp | Lighting-change tests |

## 10. Repo layout and key code

One repo, one module per responsibility from section 4, and the three trickiest pieces (ledger, state machine, camera health) sketched below so nobody starts from a blank file.

### 10.1 Repo layout

```text
trace/
  README.md            quickstart, architecture diagram, results table
  CREDITS.md           libraries, models, licences
  pyproject.toml       uv-managed
  config/              demo.yaml, production.yaml
  trace/
    capture.py         file, webcam, RTSP; drop-old-frames reader
    health.py          sharpness, brightness, texture, frozen frame
    align.py           ArUco/ORB homography, drift, TAMPERED
    people.py          segmentation, mask, visibility
    gate.py            CLAHE + SSIM change gate
    embed.py           DINOv2-small via ONNX Runtime, with cache
    verify.py          embedding + SIFT fusion, present score
    relocate.py        optional detector / feature search
    newobject.py       background model, blob logic
    stateclf.py       few-shot state classifier
    belief.py          Doubt Engine
    fsm.py             state machine, transitions, hysteresis
    policy.py          modes, schedules, authorizations, dedupe, escalation
    evidence.py        masked crops, ledger append
    ledger.py          canonical JSON, hash chain, Ed25519, verify, anchors
    outbox.py          store-and-forward, idempotent sync
    pipeline.py        the per-frame loop
    api.py             FastAPI and WebSocket
    alerts.py          Telegram, ntfy, on-screen banner
    demo_inject.py     scenario faults
  ui/                  Streamlit app and pages (Dashboard, Register, Incidents, Health, Settings, Demo)
  eval/                scenarios.yaml, ground_truth/, run_eval.py, metrics.py, baseline_naive.py
  tests/               unit and replay tests
  data/                recordings, baselines, evidence (gitignored)
```

### 10.2 Ledger: append and verify

Triggers stop accidental and casual edits; the hash chain plus off-box anchors catch a determined one (someone with file access can drop a trigger, which is exactly what the tamper demo does).

```python
# trace/ledger.py
import hashlib, json, sqlite3, time
from nacl.signing import SigningKey, VerifyKey

GENESIS = '0' * 64

def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def init(db: sqlite3.Connection) -> None:
    db.executescript('''
    CREATE TABLE IF NOT EXISTS ledger(
      idx INTEGER PRIMARY KEY AUTOINCREMENT,
      body TEXT NOT NULL, hash TEXT NOT NULL, sig TEXT NOT NULL);
    CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
      BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
    CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
      BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
    ''')

def append(db, sk: SigningKey, event: dict, crop_hashes: dict) -> str:
    row = db.execute('SELECT hash FROM ledger ORDER BY idx DESC LIMIT 1').fetchone()
    prev = row[0] if row else GENESIS
    body = canon({'event': event, 'crops': crop_hashes, 'prev': prev, 'ts_ns': time.time_ns()})
    h = sha(body)
    sig = sk.sign(h.encode()).signature.hex()
    db.execute('INSERT INTO ledger(body, hash, sig) VALUES (?,?,?)', (body.decode(), h, sig))
    db.commit()
    return h

def verify(db, vk: VerifyKey) -> dict:
    prev = GENESIS
    for idx, body, h, sig in db.execute('SELECT idx, body, hash, sig FROM ledger ORDER BY idx'):
        if sha(body.encode()) != h:
            return {'ok': False, 'bad_idx': idx, 'why': 'body does not match hash'}
        if json.loads(body)['prev'] != prev:
            return {'ok': False, 'bad_idx': idx, 'why': 'chain broken'}
        try:
            vk.verify(h.encode(), bytes.fromhex(sig))
        except Exception:
            return {'ok': False, 'bad_idx': idx, 'why': 'bad signature'}
        prev = h
    return {'ok': True, 'head': prev}

def check_anchors(db, anchors) -> dict:
    # anchors: list of (idx, head_hash) published off-box (Telegram, TSA, local file)
    for idx, head in anchors:
        r = db.execute('SELECT hash FROM ledger WHERE idx=?', (idx,)).fetchone()
        if not r or r[0] != head:
            return {'ok': False, 'bad_anchor_idx': idx}   # truncated or rewritten
    return {'ok': True}
```

Also re-hash each referenced crop file during verification and compare with `crops`, so swapping an image is caught too. A full-chain rewrite re-signs only if the attacker has the private key; keep it off the edge disk in production (TPM or HSM) and anchor the head every 60 s.

### 10.3 State machine sketch

This is the skeleton of section 6; fill in the exit hysteresis from table 6.2 where marked.

```python
# trace/fsm.py
import math
from dataclasses import dataclass, field
from enum import Enum

class S(str, Enum):
    SAFE = 'SAFE'; OCCLUDED = 'OCCLUDED'; UNKNOWN = 'UNKNOWN'; MOVED = 'MOVED'
    SWAPPED = 'SWAPPED'; STATE_CHANGED = 'STATE_CHANGED'; MISSING = 'MISSING'
    NEW_OBJECT = 'NEW_OBJECT'; TAMPERED = 'TAMPERED'; RETURNED = 'RETURNED'

@dataclass
class Obs:                    # one verification of one asset
    t: float                  # monotonic seconds
    present_score: float      # fused match score, 0..1
    drift_frac: float         # centroid drift / baseline diagonal
    visibility: float         # 0..1
    empty_match: float        # SSIM of region vs empty_roi
    camera_ok: bool
    found_elsewhere: bool

@dataclass
class AssetFSM:
    cfg: object               # thresholds, timers, half-life for this asset
    state: S = S.SAFE
    belief: float = 1.0
    t_prev: float = 0.0
    votes: dict = field(default_factory=dict)
    since: dict = field(default_factory=dict)

    def hold(self, name, cond, t, n=None, secs=None) -> bool:
        if not cond:
            self.votes[name] = 0
            self.since.pop(name, None)
            return False
        self.votes[name] = self.votes.get(name, 0) + 1
        self.since.setdefault(name, t)
        return (n is not None and self.votes[name] >= n) or (secs is not None and t - self.since[name] >= secs)

    def step(self, o: Obs) -> S:
        c = self.cfg
        dt, self.t_prev = o.t - self.t_prev, o.t
        visible = o.camera_ok and o.visibility >= c.vis_floor
        verified = visible and o.present_score >= c.present
        if verified:
            self.belief = 1.0
        else:                 # slower decay while merely occluded
            slow = 0.25 if (o.camera_ok and not visible) else 1.0
            self.belief *= math.exp(-c.lam * slow * dt)

        moved   = self.hold('moved', verified and o.drift_frac > c.move_frac, o.t, n=5)
        swapped = self.hold('swapped', visible and o.empty_match < 0.7 and o.present_score < c.present, o.t, secs=c.swap_s)
        missing = self.hold('missing', visible and o.empty_match >= 0.85 and not o.found_elsewhere, o.t, secs=c.missing_wait_s)

        if not o.camera_ok:                 # invariant: never SAFE while blind
            nxt = S.UNKNOWN if self.belief < 0.5 else S.OCCLUDED
        elif missing:  nxt = S.MISSING
        elif swapped:  nxt = S.SWAPPED
        elif moved:    nxt = S.MOVED
        elif not visible:
            nxt = S.UNKNOWN if self.belief < 0.5 else S.OCCLUDED
        elif verified:
            nxt = S.RETURNED if self.state in (S.MOVED, S.MISSING, S.SWAPPED) else S.SAFE
        else:
            nxt = S.UNKNOWN if self.belief < 0.5 else self.state
        # TODO: apply exit hysteresis (6.2) before accepting nxt, then RETURNED -> SAFE after 5 s
        self.state = nxt
        return nxt
```

### 10.4 Camera health sketch

```python
# trace/health.py
import cv2

def health(frame_bgr, base: dict) -> dict:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(g, cv2.CV_64F).var()
    bright = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[..., 2].mean() / 255.0
    edges = cv2.Canny(g, 80, 160) > 0
    h, w = edges.shape
    tiles = [edges[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4].mean() for i in range(4) for j in range(4)]
    low = sum(t < 0.3 * b for t, b in zip(tiles, base['tiles'])) / 16
    return {
        'sharp': sharp < 0.35 * base['sharp'],
        'bright': bright < 0.15,
        'texture': low > 0.6,
    }   # align.py adds the landmark check; obstructed = 2 or more fail for 1.5 s
```

### 10.5 Tests to write first

1. Tampering any ledger row, deleting a middle row, truncating the tail and swapping a crop file are each detected.
2. Invariants: an obstructed camera never yields SAFE; MISSING never fires with visibility below 0.85; no alert fires on a single frame.
3. Hysteresis: a score hovering at the threshold does not flap.
4. Policy: dedupe, cooldown, escalation timers and authorized-move windows.
5. Replay test: a short recorded clip with known events reproduces the same event list every run.

## 11. Evaluation harness and metrics

Evaluation carries 25% of the score and is the best defence in Q&A: record once, label with a script, replay deterministically, and report every number with an interval.

### 11.1 Recording plan

- **Setup:** fixed tripod, a branch-like shelf or desk corner, 6 assets, 1280×720 at 15 fps. Lock the webcam's exposure and white balance if the driver allows; auto-exposure drift causes false lighting alarms in the demo too.
- **Sessions:** A window light, B lamp only, C mixed, D dim. Record each scenario in at least two sessions.
- **Ground truth:** a run sheet with a beep-timed script, plus a small `mark.py` that logs a key press at the start and end of each action. One CSV per recording with columns `recording, t_start_s, t_end_s, asset_id, event_type, notes`.
- **Names:** `s{session}_{scenario}_{trial}.mp4` and the matching `.csv`.

| Event class | Trials | Must alert? | Applies to |
| --- | --- | --- | --- |
| Move beyond threshold | 140 | Yes | All 6 assets |
| Remove asset (MISSING) | 120 | Yes | All 6 assets |
| Swap with a look-alike | 80 | Yes | Assets with a look-alike |
| State change (open or close) | 60 | Yes | Safe door, printer cover |
| New unregistered object | 80 | Yes | Zone level |
| Lens obstruction | 60 | Yes | Camera level |
| Person walks through or stands in front | 100 | No | All assets |
| Lighting change (lamp on or off, dim) | 50 | No | All assets |
| Small nudge below threshold | 30 | No | All assets |

That is 720 labelled trials. Separate runs: 20 network-cut runs (30 to 60 s cut during events), 20 camera-bump trials (TAMPERED expected), 30 fault injections for the no-silent-failure claim, and 30 ledger tamper attempts. Where an event does not apply to an asset class, reallocate those trials to the classes where it does.

**Idle run.** At least 3 camera-days of idle footage (for example 2 cameras for 36 hours) with natural lighting change. Any alert is a false alarm unless a logged staff action explains it.

### 11.2 Matching rule

- An alert matches a ground-truth event when asset and type agree and the alert time falls in `[t_start, t_end + W]`, with W = 5 s for MOVED, SWAPPED, STATE\_CHANGED, NEW\_OBJECT and CAMERA\_OBSTRUCTED, and W = T2 + 5 s for MISSING.
- Each ground-truth event matches at most one alert (the first). An unmatched alert is a false positive; an unmatched event is a false negative; a wrong type counts as both.
- On "must not alert" trials, any MOVED, MISSING, SWAPPED or NEW\_OBJECT alert in the window is a false positive.
- **Latency** = alert time minus the marked action start, which is conservative.

### 11.3 What to compute and report

| Output | Detail |
| --- | --- |
| Precision and recall per event type | With Wilson 95% intervals (`statsmodels.stats.proportion.proportion_confint(k, n, method='wilson')`); micro and macro averages |
| Confusion matrix | Rows are ground truth, columns are predicted type including "none" |
| Latency | P50, P95 and max per event type, plus a histogram; end-to-end to the Telegram delivery time |
| False alarms | Per camera-day from the idle run; if zero, report the rule-of-three upper bound |
| Calibration | Reliability diagram and expected calibration error on dev and test |
| Robustness | Metrics split by lighting session and occlusion level |
| Resilience | Network-cut runs: events lost (target 0), maximum delivery delay; ledger verification after sync |
| Security | Tamper detection out of 30 with its interval |
| Resources | CPU, RAM, fps and per-stage latency on the demo laptop |

### 11.4 Ablation against a naive baseline

Run the same recordings through five variants and show how each idea removes errors. This is the most persuasive single slide in the final.

| Variant | What is switched on | Expected effect |
| --- | --- | --- |
| 1. Naive | Region change or disappearance for 2 s raises an alert | Many false alarms from occlusion, lighting and lens cover |
| 2. Plus temporal voting | N-frame votes and timers | Fewer flicker alerts |
| 3. Plus embedding verification | Identity match, swap detection | Fewer lighting false alarms, SWAPPED works |
| 4. Plus positive evidence and camera health | MISSING needs proof; obstruction handled | Occlusion and lens cover stop reading as theft |
| 5. Full TRACE | Adds the policy layer | Legitimate moves stop alerting |

### 11.5 Reproducibility and hygiene

- **Split by recording session, never by frame:** about 40% of sessions for tuning, 60% for the reported test. Freeze thresholds before running the test set.
- Log the git commit hash, config file and library versions with every run.
- One command rebuilds the report: `python -m eval.run_eval --recordings data/rec --out report/`, which writes `report.md`, figures and the raw predictions CSV.
- Recordings that show team members need their consent; keep them local, and use mock assets only.

## 12. Round 2 demo plan and Q&A defense

The slot is a 10-minute pitch and demo plus Q&A: one minute of problem, a fixed-order live demo, two minutes of results, and a rehearsed answer for the panel's live twist, with a recorded fallback behind it.

### 12.1 Ten-minute run sheet

| Time | Segment | What happens |
| --- | --- | --- |
| 0:00 to 1:00 | Hook | "A safe moved at 2 AM, the camera was blocked, and the log was edited." The three gaps and the idea in one line. |
| 1:00 to 2:00 | Registration | Register one asset live in the wizard (the others are pre-registered); show the baseline and the dry run. |
| 2:00 to 3:30 | Core events | Move the safe: MOVED, phone buzzes, before and after crops. Remove the laptop: MISSING after the wait. Place an unknown box: NEW\_OBJECT. |
| 3:30 to 4:30 | Honest uncertainty | Hand in front of an asset: OCCLUDED, no alarm. Cover the lens: CAMERA\_OBSTRUCTED, assets decay to UNKNOWN. "It never lies." |
| 4:30 to 5:30 | Network loss | Cut the network: local alerts continue, queue badge rises; reconnect: "delayed" badge, zero events lost. |
| 5:30 to 6:30 | Evidence and tamper | Open the incident; run Tamper ledger; Verify ledger turns red at the edited row; restore. |
| 6:30 to 8:30 | Results | Confusion matrix, intervals, latency, the ablation slide; open the evidence folder to show no faces. |
| 8:30 to 9:30 | Scale and cost | Production architecture, estimated cost per branch, roadmap. |
| 9:30 to 10:00 | Close | "It proves, with evidence and honest confidence, what it knows." Then Q&A. |

### 12.2 Twist playbook

Rule for every twist: narrate what the system believes and why; never claim certainty.

| Likely twist | Response | What the panel sees |
| --- | --- | --- |
| Block or cover the camera | Do nothing; point at the screen | CAMERA\_OBSTRUCTED in about 2 s, assets go UNKNOWN, no theft alarm |
| Turn the lights off | Same as above | Darkness fails the brightness check: blind, not stolen |
| Cut the network | Show the banner and queue depth, then reconnect | Local alert continues; synced events carry a "delayed" badge; ledger verifies |
| Put an object in front of an asset | Wait | OCCLUDED with slow belief decay; MISSING only after the view clears and the wait passes |
| Swap the asset for a similar one | Open the incident | SWAPPED with side-by-side crops |
| Knock the camera | Show the drift value | TAMPERED and a re-anchor prompt |
| Something unplanned | Say what the state and confidence are and why | Honest behaviour under uncertainty, which is the brand |

### 12.3 Roles

Four people: **Presenter** (story and results), **Operator** (laptop and dashboard), **Props** (moves, covers, swaps on cue), **Q&A lead** (technical answers). With three, merge Props into Operator; with two, the Operator also owns props.

### 12.4 Checklist, 60 minutes before

- Power: laptop plugged in, high-performance mode, notifications off, screen resolution set for the projector.
- Camera: tape marks on the desk for tripod position; exposure and white balance locked; ArUco stickers visible.
- **Re-register every asset in the venue's lighting** and run 10 minutes idle before going on; baselines from your room will misfire under different light.
- Network: phone on hotspot, Telegram and ntfy test messages received, router or hotspot ready to cut.
- Data: clean demo database seeded, ledger anchors file present, fallback video on the desktop, slides open.

### 12.5 Fallbacks

| Failure | Fallback |
| --- | --- |
| Webcam fails | Play the recorded clip through the same pipeline (file source) |
| Laptop fails | Backup screen recording of a clean 3-minute run, plus the slides |
| Telegram blocked | ntfy push and the on-screen banner |
| Model download needed | Weights are already cached locally (section 9.2) |

### 12.6 Likely questions and answers

| Question | Answer |
| --- | --- |
| How do you reach 99.99%? | We do not claim it; proving it needs about 30,000 clean trials. We report intervals, guarantee no silent failures, and send uncertain cases to review. |
| How does it find a safe with no training? | Region matching against a Visual DNA from 3 photos. A detector is used only to re-find moved assets or new objects; the state classifier is few-shot. |
| Won't normal office use spam alerts? | The policy layer (fixed vs. mobile, business hours, authorized move, dedupe, escalation) and a measured false-alarm rate per camera per day. |
| Why not just run YOLO? | COCO has no safe, printer or gold-machine class, and a detector cannot tell whether it is still the same object. Our ablation shows what the extra logic removes. |
| Camera blocked or network down? | Health checks raise CAMERA\_OBSTRUCTED, assets go UNKNOWN, the edge keeps alerting locally and syncs later. |
| What if the log is rewritten? | Entries are signed and hash-chained; the head is anchored off-box, so truncation or rewrite is detected; in production the key lives off the edge disk. |
| Privacy? | People are masked in memory; only asset crops are stored; no face recognition; a retention policy; aligned with the DPDP Act 2023 (confirm with legal). |
| Identical printers? | A small ArUco or QR tag per asset; without a tag this is a stated limitation. |
| Lighting changes? | CLAHE, multi-lighting baselines, calibrated thresholds, and tested across four lighting sessions. |
| Camera knocked? | Landmark drift raises TAMPERED and a re-anchor prompt. |
| Hardware and speed? | CPU only; about 5 fps processing on a laptop; production target is a Pi 5, N100 mini-PC or Jetson. |
| Scale to hundreds of branches? | Per-branch edge nodes, MQTT, a central fleet dashboard, OTA config and model bundles, per-branch signing keys. |
| Cost? | Software is free; hardware is about ₹25-40k per branch (an estimate: edge box, 2 IP cameras, UPS). |
| Can someone spoof it with a photo? | Tags, temporal-noise and parallax checks help; a high-quality spoof on one camera could still fool it, and a second camera closes that gap. |
| What can't it do? | Assets inside closed drawers, very low light, unlabelled look-alikes, and long occlusions, which end in UNKNOWN by design rather than a guess. |
| What is different from commercial video analytics? | The positive-evidence rule, doubt-aware states, privacy-first evidence and a signed ledger, all open source, focused on assets rather than people. |
| What did you build vs reuse? | Reused OpenCV, DINOv2, MediaPipe and standard libraries (see CREDITS); built the pipeline, state machine, policy, ledger, UI and evaluation. |
| Is the false-positive feedback safe? | Threshold changes are bounded, logged and reviewable. |
| How does this fit L&T Finance branches? | Safes, gold valuation machines and printers, after-hours monitoring, audit-grade evidence, and webhooks into ticketing or SIEM tools. |

## 13. Build plan, roles and risks

Four days remain after the Round 1 submission and four workstreams share them; the critical path is the per-asset verifier and the state machine, so start those first and keep everything else behind a fake until they work.

### 13.1 Workstreams

| Workstream | Owns | First deliverable |
| --- | --- | --- |
| A. Vision | `capture`, `health`, `align`, `people`, `gate`, `embed`, `verify` | Registered asset reads SAFE on a recorded clip |
| B. Logic | `fsm`, `belief`, `policy`, `newobject`, `stateclf`, optional `relocate` | Scripted observations drive every state in the tests |
| C. Platform | `ledger`, `evidence`, `outbox`, `api`, `alerts` | Event in, signed ledger row, Telegram message out |
| D. Product and proof | `ui`, `demo_inject`, `eval`, deck, Q&A | Dashboard shows live states from fakes; recording script ready |

Agree the `Obs` dataclass, the event JSON and the API (sections 7 and 10) in the first hour; everyone builds against fakes until the real parts land. With fewer than four people, merge B into A and C into D.

### 13.2 Milestones and gates

The drawing below shows what must be true at each gate. If a gate slips, cut scope in the order listed in 13.3 rather than moving the final.

&#91;embedded content: build plan · four workstreams, five days, five gates\]

Monday and Tuesday carry the critical path; Wednesday's recordings feed Thursday's evaluation, and nothing new is built after Thursday 21:00.

### 13.3 Scope tiers

| Tier | Items |
| --- | --- |
| Must (the demo fails without these) | Registration and baselines, verifier, MOVED, MISSING with empty-region proof, NEW\_OBJECT, CAMERA\_OBSTRUCTED, network-loss outbox with SYSTEM\_OFFLINE, UNKNOWN and the Doubt Engine, ledger with Verify and the tamper demo, masked evidence crops, Telegram alert, dashboard (status, incident, log), demo panel, evaluation report with intervals |
| Should | SWAPPED with tags, STATE\_CHANGED for the safe door, authorized-move policy, people-mask polish, ablation slide |
| Could | Relocator or detector, ArUco landmarks and TAMPERED, review queue, RFC 3161 anchor, floor-plan homography, settings page |
| Roadmap only | MinIO, Keycloak, MQTT, Prometheus, ESP32 sensor, two-camera fusion, VLM verifier, OTA updates |

Cut from the bottom of Should and Could first, never from Must.

### 13.4 Working rules

- Two stand-ups a day (morning and evening), 10 minutes each.
- `main` always runs `make demo`; merge nightly and tag demo-ready commits.
- Code freeze Thursday 15 Oct, 21:00. Friday morning is for venue setup and rehearsal only.
- Record yourself running the full demo once on Thursday as the backup video.

### 13.5 Risk register

| Risk | Likelihood and impact | Mitigation |
| --- | --- | --- |
| Webcam auto-exposure flicker causes false alarms | High, high | Lock exposure and white balance, CLAHE in the gate, multi-lighting baselines, test at the venue |
| Lighting at the venue differs from your room | High, high | Re-register every asset on site 60 minutes before |
| Plain cardboard assets have no features | High, medium | ArUco tags on assets; embedding-only mode |
| Time overrun | High, high | Scope tiers, code freeze, always-demoable `main` |
| DINOv2 too slow on CPU | Medium, medium | ONNX Runtime, 224 px crops, cache unchanged regions, once-per-second identity checks; fallback to ORB plus SSIM |
| Telegram or model hosts blocked on campus Wi-Fi | Medium, high | Pre-cached weights, ntfy fallback, mobile hotspot |
| The panel's twist is something unplanned | Medium, medium | Narrate state and confidence; UNKNOWN is a valid, honest answer |
| Evaluation comes out below targets | Medium, medium | Report it with intervals, show the ablation and the fix, never tune on the test set |
| Idle run shows false alarms | Medium, medium | Diagnose, fix, re-run a partial window, and report both runs |
| Streamlit refresh flicker | Medium, low | `st.fragment`, MJPEG stream, htmx fallback |
| A team member is unavailable | Low, high | Pair on the critical path, keep interfaces documented |
| Demo laptop fails | Low, high | Backup video and a second laptop with the repo and weights |

## 14. Production roadmap, enterprise concerns and submission checklist

Round 1 is won by a clean 5-slide file submitted on time; the roadmap and the enterprise table below are what you say when the panel asks "what next?" and "is this safe to deploy?".

### 14.1 Submission checklist (before 9 AM, 12 Oct)

- [ ] Fill 2 to 4 real team names on slide 1 (they are still `[Name 1]` to `[Name 4]`).
- [ ] Confirm the team is registered ([Registration Form](https://forms.cloud.microsoft/r/dTtn2mNJsT); the deadline was 11 Oct).
- [ ] Keep the ₹25-40k cost line only if you are comfortable defending it as an estimate.
- [ ] Open the file in PowerPoint itself (not only a preview): check Calibri renders, nothing overflows, speaker notes are present.
- [ ] Exactly 5 slides, saved as `.pptx` (the brief says PPT only), for example `TRACE_IntelliWatch_Round1.pptx`.
- [ ] Read every slide aloud once for typos; check the numbers match this doc (720 events, targets, state names).
- [ ] No real customer data and no personal photos anywhere in the deck.
- [ ] Upload through the [Submission Form](https://forms.cloud.microsoft/r/8g2dbC0Rkz) and save the confirmation; keep a local copy.

### 14.2 Production roadmap

1. **Pilot, one branch, two cameras, 4 to 6 weeks.** Measure the false-alarm rate on a real layout (with the branch's consent and legal sign-off), tune per-class thresholds and half-lives.
2. **Five to ten branches.** Fleet dashboard, MQTT transport, PostgreSQL and object-lock storage, SSO and roles, signed OTA bundles for models and config.
3. **All branches.** Ticketing and SIEM integration, SLA and health monitoring, drift monitoring from the false-positive feedback, per-branch signing keys in a TPM or HSM, two-camera fusion for vaults and gold counters.
4. **Extensions.** Tilt or reed sensors and RFID cross-checks on safes, a local VLM verifier for ambiguous cases, asset-utilisation analytics.

### 14.3 Enterprise concerns and how TRACE answers them

| Concern | Approach |
| --- | --- |
| Camera tampering | Health checks, landmark drift, TAMPERED and CAMERA\_OBSTRUCTED states |
| Photo or screen spoofing | Asset tags, temporal-noise and parallax checks, second camera for high-value assets |
| Insider log deletion | Append-only ledger, hash chain, signatures, off-box anchors |
| Edge box compromise | Signed images, signing key off the edge disk, least-privilege service accounts |
| Alert-channel failure | Multi-channel alerts, local banner or siren, escalation timers |
| Data leakage | People masked in memory, asset crops only, disk encryption, retention limits, no face recognition |
| Model drift | False-positive feedback with bounded threshold changes, scheduled re-baselining |
| Supply chain | Pinned dependencies, SBOM, CREDITS file |
| Compliance | Purpose limitation, notice signage and retention aligned with the DPDP Act 2023; confirm with legal before any real deployment |
| Integration | Webhooks, syslog or CEF to a SIEM, tickets in the existing tool |
| Operations | Health endpoint, metrics, runbooks, remote config |

*Not looked up for this doc:* library licences and version numbers are from general knowledge. Confirm each on the project's own repository before relying on it.
