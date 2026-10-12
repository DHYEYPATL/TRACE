# TRACE Architectural Restructuring Blueprint

## Scope
- Modular monolith in Python (OpenCV, FastAPI, SQLite)
- Single fixed, calibrated camera covering registered physical assets (Safe, Printer, Laptop)
- Distinction between visual evidence, camera state, asset observation, and incident lifecycle
- Zero synthetic fallbacks in production/real camera mode
- Real frame envelopes with monotonic timing and frame deduplication

## Core Architecture Pillars

1. **Capture Service & Frame Envelope**:
   - `FrameEnvelope` with `camera_id`, `frame_id`, `captured_at_utc`, `captured_monotonic_ns`, `image_bgr`, `source_status`.
   - Frame deduplication: a cached or re-read frame is processed at most once; never advances temporal voting.

2. **Camera Alignment & Coordinate Systems**:
   - Canonical reference coordinate system based on the reference frame.
   - Warps current frame into reference coordinates: `aligned_frame = cv2.warpPerspective(current_frame, H, (w, h))`.
   - Regions defined in reference coordinates are directly evaluated on `aligned_frame`.
   - Physical movement of the asset is measured by locating the asset (e.g. via template / relocator) and computing distance from expected reference centroid.
   - Camera shift is detected by landmark displacement and homography drift; does NOT trigger false asset movement!

3. **Visual Observation Engine**:
   - Typed `AssetObservation`: `presence_score`, `identity_score`, `visibility_score`, `expected_location`, `observed_location`, `location_status`, `observation_status`.
   - No fake uniform-grey empty baselines (`np.full_like(crop, 160)` removed).
   - Real ArUco / visual tag detection: if tag expected, actually detect it or mark UNCERTAIN.
   - Safe door state classifier: require genuine enrolled samples or mark door monitoring unsupported (no synthetic drawings).

4. **Separated State Models**:
   - **Camera State**: `STARTING`, `CALIBRATING`, `HEALTHY`, `STALE`, `OBSTRUCTED`, `ALIGNMENT_LOST`, `OFFLINE`, `RECOVERING`.
   - **Asset Observation State**: `PRESENT_EXPECTED`, `RELOCATED`, `NOT_OBSERVED`, `OCCLUDED`, `IDENTITY_UNCERTAIN`, `UNKNOWN`.
   - **Incident State**: `CANDIDATE`, `CONFIRMED`, `NEEDS_REVIEW`, `ALERTED`, `ACKNOWLEDGED`, `RESOLVED`.
   - Camera outages produce one camera incident, not an incident per asset.
   - Non-detection is reported as `NOT_OBSERVED`, never assuming "theft".

5. **Tracked New Object Lifecycle**:
   - `FIRST_SEEN -> TRACKED -> CONFIRMED -> INCIDENT_OPEN -> RESOLVED`.
   - Enforce 1-to-1 track association.
   - Emit incident only once when crossing confirmation threshold; update existing incident on subsequent frames.

6. **Transactional Persistence & Asia/Kolkata Business Hours**:
   - Atomic transactions across ledger append (`commit=False`), events, incidents, and outbox enqueue.
   - Explicit timezone handling using `ZoneInfo("Asia/Kolkata")`.
