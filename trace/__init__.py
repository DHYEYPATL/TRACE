"""
TRACE: Trusted Real-time Asset Confidence Engine
A software-only, privacy-first asset confidence engine for branch monitoring.
"""

from __future__ import annotations

# Top-level module aliases matching Blueprint Section 10 layout
from trace.vision import capture
from trace.vision import health
from trace.vision import align
from trace.vision import people
from trace.vision import gate
from trace.vision import embed
from trace.vision import verify
from trace.vision import relocate
from trace.vision import newobject
from trace.vision import stateclf

from trace.logic import belief
from trace.logic import fsm
from trace.logic import policy

from trace.platform import db
from trace.platform import ledger
from trace.platform import evidence
from trace.platform import outbox
from trace.platform import alerts
from trace.platform import demo_inject
from trace.platform import anchor

from trace import pipeline
from trace import config

__all__ = [
    "capture",
    "health",
    "align",
    "people",
    "gate",
    "embed",
    "verify",
    "relocate",
    "newobject",
    "stateclf",
    "belief",
    "fsm",
    "policy",
    "db",
    "ledger",
    "evidence",
    "outbox",
    "alerts",
    "demo_inject",
    "anchor",
    "pipeline",
    "config",
]

__version__ = "1.0.0"
