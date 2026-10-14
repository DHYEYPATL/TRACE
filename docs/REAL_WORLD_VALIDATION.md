# AssetWatch AI — Real-World Validation Checklist

Run this with the actual USB camera, desk/shelf, lighting and physical items you intend to monitor. Unit tests cover the logic; only the following integration checks confirm real-camera behaviour.

---

## A. Setup and compute

1. Run `scripts/check_setup.py --probe-cameras` while the app is closed.
2. Start AssetWatch (`Run-AssetWatch.bat`). In the dashboard, confirm **Camera 1 (USB webcam)** is selected. If the laptop camera appears instead, switch to index 1 or 2 and click **Reconnect**.
3. Confirm **CAMERA FPS** and **AI INFERENCE** both show non-zero values. **COMPUTE** should display the NVIDIA GPU name if CUDA is active.
4. Confirm the monitoring mode badge in the top bar shows **⚡ Always on** (default). No **IDLE** badge should appear.

---

## B. Registration and coverage meter

1. Point the USB webcam at a physical object (bottle, laptop, phone, keys).
2. Click the amber detection box. Enter a name, click **Register selected item**.
3. The coverage meter under the asset card should show **Gallery 1/10 · Add reference views for better tracking** in red.
4. Click **＋ Reference view** then select the object from a different angle (±45°, side, top-down). Repeat until you have 4–6 views. The meter should turn amber then green.
5. Test that adding a crop of a *different* object is rejected with an error message.

---

## C. Identity robustness (viewpoint changes)

1. Register an object with at least 4 reference angles.
2. Rotate it 45° in place. Confirm it stays **PRESENT** with a match score visible.
3. Tilt it toward the camera. Confirm it still matches (partial-view handling).
4. Cover it for 5 seconds. Confirm **CHECKING** state, then **PRESENT** on uncovering.
5. Remove it entirely for 30+ seconds. Confirm **MISSING** event appears in Audit Trail.
6. Return it. Confirm the **↩ Recovered** badge and a cyan bounding box flash appear.

---

## D. Two identical objects (the hard test)

1. Register **Object A** and **Object B** (two visually identical bottles/cups).
2. Add 3+ reference angles per object.
3. Move them simultaneously so they cross paths.
4. Confirm names stay attached to the correct physical items.
5. If an assignment is ambiguous the system leaves both amber — this is correct behaviour, not a bug.

---

## E. Monitoring modes

1. Switch to **🌙 Night (idle)** in the mode drop-down. Confirm the **IDLE** badge appears in the top bar and inference FPS drops to ~2.
2. Switch back to **⚡ Always on**. Confirm full-speed inference resumes.
3. Restart the app. Confirm the last-set mode is restored (persisted in SQLite).

---

## F. Hard delete

1. Archive an asset using **Archive** — confirm it disappears from the dashboard but events remain in history.
2. Register another asset, then click **🗑** (hard delete). Confirm a confirmation dialog appears.
3. After confirmation: asset gone, events gone, crop folder under `data/references/<id>` deleted.

---

## G. Camera health gate

1. Disconnect the USB webcam while an object is registered as PRESENT.
2. Confirm the camera dot turns red and the warning banner appears.
3. Confirm **no MISSING event is created** while the camera is unhealthy.
4. Reconnect the camera. Confirm it reconnects automatically and monitoring resumes.

---

## H. CSRT manual tracker

1. Draw a box around an object the detector does not recognise (e.g. a pen or cable).
2. Register it. The tracker mode should show **Manual tracker**.
3. Move the object slowly. Confirm the purple box follows it.
4. When YOLO later detects a matching object near the CSRT box, confirm it is **PROMOTED** to AI detection (event in Audit Trail, box turns green).

---

## I. Performance budget

Record these values after ~5 minutes of operation:

| Metric | Target | Observed |
|---|---|---|
| Camera FPS | ≥ 20 | |
| AI Inference FPS | ≥ 5 | |
| Inference latency | ≤ 130 ms | |
| GPU name shown | RTX 3050 Ti | |
| Missing false-alarm (camera unplugged) | None | |

---

## Known limits (do not over-claim)

- Two visually identical, unmarked items of the same class **cannot** be reliably distinguished if no visible difference exists — the system rejects the ambiguous match rather than guessing.
- CSRT loses the tracked object after full occlusion or when it leaves the frame; manual re-selection is needed.
- Movement alerts are relative to the registered anchor position — re-register after moving the camera.
- This is a local laptop monitoring tool, not a certified security system.
