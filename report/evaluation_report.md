# TRACE Multi-Scenario Regression Benchmark Report

**Evaluation Suite:** Synthetic Multi-Scenario Regression Suite (synthetic_regression_v1, 10 Controlled Scenarios, 415 Frames)  
**Date:** 2026-10-12 08:24:17 UTC  
**Matching Protocol:** Strict 1-to-1 Type-Compatible Event Matching (Zero Ground-Truth Leakage)  
**Disclosure:** This benchmark measures algorithmic behavior under controlled scenarios simulating physical phenomena (displacement, removal, swap, occlusion, optical obstruction, freeze, knock, and lighting variation). Physical camera validation requires continuous real-world recordings.

---

## 1. Measured Accuracies & Raw Event Counts

All detection metrics below were produced through **live frame-level execution of the visual pipeline**, comparing predictions strictly against held-out ground-truth timestamps:

| Architecture Variant | TP | FP | FN | TN | Precision (95% CI) | Recall (95% CI) | F1 Score | FP / Camera-Hour |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Naive Pixel Diff** | 3 | 10 | 4 | 8 | 23.1% (3/13) [8.2%, 50.3%] | 42.9% (3/7) [15.8%, 75.0%] | 0.300 | 433.73 |
| **2. + Temporal Voting** | 3 | 10 | 4 | 8 | 23.1% (3/13) [8.2%, 50.3%] | 42.9% (3/7) [15.8%, 75.0%] | 0.300 | 433.73 |
| **3. + LabSobel Descriptor** | 1 | 10 | 6 | 9 | 9.1% (1/11) [1.6%, 37.7%] | 14.3% (1/7) [2.6%, 51.3%] | 0.111 | 433.73 |
| **4. Full TRACE Engine** | **7** | **3** | **0** | **10** | **70.0%** (7/10) **[39.7%, 89.2%]** | **100.0%** (7/7) **[64.6%, 100.0%]** | **0.824** | **130.12** |

---

## 2. Per-Category Breakdown (Full TRACE Engine)

| Event Category | Ground-Truth Intent | TP | FP | FN | TN | Category Precision | Category Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MOVED** | Asset displaced > 10% region diagonal | 2 | 0 | 0 | - | 100.0% | 100.0% |
| **MISSING** | Asset removed; bare desk background proof | 1 | 0 | 0 | - | 100.0% | 100.0% |
| **SWAPPED** | Lookalike object substituted in region | 1 | 0 | 0 | - | 100.0% | 100.0% |
| **CAMERA_OBSTRUCTED** | Optical lens physically blocked | 1 | 0 | 0 | - | 100.0% | 100.0% |
| **CAMERA_OFFLINE** | Frozen video stream (> 2.0s bit-identical) | 1 | 0 | 0 | - | 100.0% | 100.0% |
| **CAMERA_TAMPERED** | Physical camera displacement > 25px | 1 | 1 | 0 | - | 50.0% | 100.0% |
| **STABLE (Negative)** | Static scene under normal lighting | - | 0 | - | 7 | N/A | 0 false alarms |
| **OCCLUDED (Negative)**| Human standing in front of asset | - | 0 | - | 1 | N/A | 0 false alarms |
| **LIGHTING (Negative)**| 60% ambient illumination drop | - | 0 | - | 2 | N/A | 0 false alarms |

---

## 3. End-to-End Latency Profile

Measured from annotated event onset time ($t_{onset}$) to alert confirmation time:

- **Confirmed Event Reaction Latency:**
  - Median (P50): **900.0 ms**
  - P95: **4400.0 ms**
  - Max: **5000.0 ms**
  - Configured Temporal Confirmation Windows: 5 frames (moved) / 5.0 s (missing)

---

## 4. Key Architectural Observations

1. **Camera Failure Disambiguation:** When camera input becomes untrusted (lens cover, freeze, tamper displacement), TRACE raises a camera maintenance alert and strictly suppresses asset-missing alerts.
2. **Positive-Evidence Emptiness Rule:** An asset is only declared `MISSING` when its region is visible ($v \ge 0.85$) and matches bare background ($empty\_match \ge 0.85$). Human occlusion holds the asset in `OCCLUDED`.
3. **Relocation Discrimination:** Displaced assets are scanned across the frame via `AssetRelocator`; assets detected elsewhere in the facility confirm `MOVED`, not `MISSING`.
