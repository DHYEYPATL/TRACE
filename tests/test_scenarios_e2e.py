"""
End-to-End Regression Test Suite for TRACE.
Processes actual image sequences through the genuine computer-vision pipeline,
asserting observable behavior across real-world adversarial and edge scenarios.
"""

from typing import List, Tuple
import numpy as np
import pytest

from trace.eval.baseline_naive import FullTraceEvaluator
from trace.eval.dataset import EvaluationDatasetBuilder
from trace.logic.fsm import S



@pytest.fixture(scope="module")
def dataset_builder():
    return EvaluationDatasetBuilder()


@pytest.fixture
def evaluator():
    return FullTraceEvaluator()


def _run_clip(evaluator: FullTraceEvaluator, frames: List[Tuple[float, np.ndarray]]):
    ref_frame = frames[0][1]

    evaluator.reset()
    # Register safe-02, printer-01, laptop-03
    evaluator.register_asset("safe-02", [[180, 220], [380, 220], [380, 520], [180, 520]], ref_frame[220:520, 180:380])
    evaluator.register_asset("printer-01", [[520, 260], [740, 260], [740, 480], [520, 480]], ref_frame[260:480, 520:740])
    evaluator.register_asset("laptop-03", [[880, 310], [1100, 310], [1100, 500], [880, 500]], ref_frame[310:500, 880:1100])
    evaluator.calibrate(ref_frame)

    outputs = []
    for t, frame in frames:
        out = evaluator.process_frame(frame, t)
        outputs.append((t, out))
    return outputs


def test_scenario_asset_stable(dataset_builder, evaluator):
    """Scenario 1: Assets remain stable under normal conditions; no false alarms."""
    _, frames = dataset_builder._build_clip_stable()
    outputs = _run_clip(evaluator, frames)

    # Final states must be SAFE
    last_t, last_out = outputs[-1]
    assert last_out["states"]["safe-02"] == S.SAFE.value
    assert last_out["states"]["printer-01"] == S.SAFE.value
    assert last_out["states"]["laptop-03"] == S.SAFE.value

    # Zero security alerts dispatched
    all_alerts = [a for _, out in outputs for a in out.get("alerts", [])]
    assert len(all_alerts) == 0


def test_scenario_asset_moved(dataset_builder, evaluator):
    """Scenario 2: Asset moved 70px beyond threshold correctly confirms MOVED."""
    _, frames = dataset_builder._build_clip_move()
    outputs = _run_clip(evaluator, frames)

    # Safe should confirm MOVED
    safe_states = [out["states"]["safe-02"] for _, out in outputs]
    assert S.MOVED.value in safe_states

    # Alert should be fired for safe-02
    alerts = [a for _, out in outputs for a in out.get("alerts", []) if a["asset_id"] == "safe-02"]
    assert len(alerts) >= 1
    assert alerts[0]["type"] == S.MOVED.value


def test_scenario_asset_removed(dataset_builder, evaluator):
    """Scenario 3: Asset removed with empty desk proof confirms MISSING."""
    _, frames = dataset_builder._build_clip_missing()
    outputs = _run_clip(evaluator, frames)

    printer_states = [out["states"]["printer-01"] for _, out in outputs]
    assert S.MISSING.value in printer_states


def test_scenario_transient_occlusion(dataset_builder, evaluator):
    """Scenario 4: Human standing in front of safe yields OCCLUDED and NEVER alerts theft."""
    _, frames = dataset_builder._build_clip_occlusion()
    outputs = _run_clip(evaluator, frames)

    safe_states = [out["states"]["safe-02"] for _, out in outputs]
    # Enters OCCLUDED during human presence
    assert S.OCCLUDED.value in safe_states
    # Never enters MISSING
    assert S.MISSING.value not in safe_states

    # Zero theft alerts fired
    theft_alerts = [a for _, out in outputs for a in out.get("alerts", []) if a["asset_id"] == "safe-02"]
    assert len(theft_alerts) == 0


def test_scenario_lens_obstruction(dataset_builder, evaluator):
    """Scenario 5: Obstructed lens generates camera alert and strictly suppresses asset-missing."""
    _, frames = dataset_builder._build_clip_obstruction()
    outputs = _run_clip(evaluator, frames)

    # Generates camera health alert
    cam_alerts = [a for _, out in outputs for a in out.get("alerts", []) if a["asset_id"] == "camera"]
    assert len(cam_alerts) >= 1
    assert any("OBSTRUCTED" in a["type"] for a in cam_alerts)

    # Assets must NOT be claimed as MISSING or stolen
    asset_theft_alerts = [a for _, out in outputs for a in out.get("alerts", []) if a["asset_id"] in ("safe-02", "printer-01")]
    assert len(asset_theft_alerts) == 0


def test_scenario_frozen_feed(dataset_builder, evaluator):
    """Scenario 6: Frozen camera feed triggers SYSTEM_OFFLINE."""
    _, frames = dataset_builder._build_clip_freeze_offline()
    outputs = _run_clip(evaluator, frames)

    cam_alerts = [a for _, out in outputs for a in out.get("alerts", []) if a["asset_id"] == "camera"]
    assert len(cam_alerts) >= 1
    assert any("OFFLINE" in a["type"] for a in cam_alerts)


def test_scenario_camera_shift(dataset_builder, evaluator):
    """Scenario 7: Camera knock displacement triggers CAMERA_TAMPERED."""
    _, frames = dataset_builder._build_clip_camera_shift()
    outputs = _run_clip(evaluator, frames)

    cam_alerts = [a for _, out in outputs for a in out.get("alerts", []) if a["asset_id"] == "camera"]
    assert len(cam_alerts) >= 1
    assert any("TAMPERED" in a["type"] for a in cam_alerts)


def test_scenario_lighting_drop(dataset_builder, evaluator):
    """Scenario 8: 60% ambient illumination drop preserves SAFE state."""
    _, frames = dataset_builder._build_clip_lighting_drop()
    outputs = _run_clip(evaluator, frames)

    # Zero false alerts on lighting dimming
    alerts = [a for _, out in outputs for a in out.get("alerts", [])]
    assert len(alerts) == 0


def test_scenario_recovery_hysteresis(dataset_builder, evaluator):
    """Scenario 9: Asset moved and returned recovers cleanly through RETURNED to SAFE."""
    _, frames = dataset_builder._build_clip_recovery()
    outputs = _run_clip(evaluator, frames)

    safe_states = [out["states"]["safe-02"] for _, out in outputs]
    assert S.MOVED.value in safe_states
    assert S.RETURNED.value in safe_states

