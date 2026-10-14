import numpy as np

from assetwatch.vision import VisionEngine


def sample(embedding, histogram, orb_b64="", orb_rows=0):
    vec = np.asarray(embedding, dtype=np.float32)
    vec /= max(float(np.linalg.norm(vec)), 1e-8)
    return {"embedding": vec.tolist(), "histogram": histogram, "orb_b64": orb_b64, "orb_rows": orb_rows}


def test_identical_features_score_high():
    feature = sample([1, 0, 0], [0.6, 0.3, 0.1])
    assert VisionEngine.compare_features(feature, feature) > 0.95


def test_orthogonal_features_do_not_look_like_a_confident_identity():
    a = sample([1, 0, 0], [1, 0, 0])
    b = sample([0, 1, 0], [0, 0, 1])
    assert VisionEngine.compare_features(a, b) < 0.65


def test_best_match_refuses_ambiguous_ties():
    q = sample([1, 0, 0], [0.5, 0.4, 0.1])
    assets = [
        {"asset_id": "A", "active": True, "tracker_mode": "detector", "profile": {"samples": [q]}},
        {"asset_id": "B", "active": True, "tracker_mode": "detector", "profile": {"samples": [q]}},
    ]
    match, score = VisionEngine().best_asset_match(q, assets, min_score=0.70, min_margin=0.06)
    assert match is None
    assert score is not None and score > 0.95


def test_best_match_can_choose_a_clear_candidate():
    q = sample([1, 0, 0], [0.8, 0.15, 0.05])
    poor = sample([0, 1, 0], [0.0, 0.1, 0.9])
    assets = [
        {"asset_id": "A", "active": True, "tracker_mode": "detector", "profile": {"samples": [q]}},
        {"asset_id": "B", "active": True, "tracker_mode": "detector", "profile": {"samples": [poor]}},
    ]
    match, score = VisionEngine().best_asset_match(q, assets, min_score=0.70, min_margin=0.06)
    assert match == "A"
    assert score is not None and score > 0.95
