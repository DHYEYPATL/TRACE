"""
Dataset Manifest module for TRACE evaluation.
Provides typed definitions for ground-truth event intervals, clip specifications,
and manifest persistence. Decoupled strictly from prediction generation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EventInterval:
    """Ground truth event interval on a timeline."""
    event_type: str        # 'STABLE', 'MOVED', 'MISSING', 'OCCLUDED', 'CAMERA_OBSTRUCTED', 'CAMERA_OFFLINE', 'CAMERA_TAMPERED', 'SWAPPED', 'LIGHTING_CHANGE'
    asset_id: str          # Asset ID or 'camera' for sensor events
    start_s: float         # Onset timestamp in seconds
    end_s: float           # End timestamp in seconds
    must_alert: bool       # Whether this event should produce a high-severity security alert
    expected_state: str    # Expected final/interim state (e.g. MOVED, OCCLUDED, SAFE, etc.)
    notes: str = ""

    def contains(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s


@dataclass
class ClipManifest:
    """Ground truth specification for a single test video clip or sequence."""
    clip_id: str
    description: str
    fps: float
    duration_s: float
    events: List[EventInterval] = field(default_factory=list)
    camera_id: str = "cam-1"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationDatasetManifest:
    """Master dataset manifest containing multiple evaluation clips."""
    dataset_version: str
    description: str
    clips: List[ClipManifest] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationDatasetManifest:
        clips = []
        for c in data.get("clips", []):
            events = [EventInterval(**e) for e in c.get("events", [])]
            clips.append(ClipManifest(
                clip_id=c["clip_id"],
                description=c.get("description", ""),
                fps=c.get("fps", 5.0),
                duration_s=c.get("duration_s", 0.0),
                events=events,
                camera_id=c.get("camera_id", "cam-1"),
                meta=c.get("meta", {}),
            ))
        return cls(
            dataset_version=data.get("dataset_version", "1.0.0"),
            description=data.get("description", ""),
            clips=clips,
        )

    def save(self, filepath: Path | str) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filepath: Path | str) -> EvaluationDatasetManifest:
        p = Path(filepath)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)
