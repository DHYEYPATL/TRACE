"""
Configuration module for TRACE.
Loads YAML files, resolves environment variables, and parses into typed Pydantic models.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


def _env_subst(match: re.Match) -> str:
    var_name = match.group(1)
    default_val = match.group(2) if match.group(2) is not None else ""
    return os.environ.get(var_name, default_val)


def expand_env_vars(text: str) -> str:
    """Replaces ${VAR} or ${VAR:default} with environment values."""
    pattern = re.compile(r"\${([A-Za-z0-9_]+)(?::([^}]*))?}")
    return pattern.sub(_env_subst, text)


class AssetThresholds(BaseModel):
    move_frac: float = 0.10
    present: float = 0.70
    vis_floor: float = 0.85


class AssetConfig(BaseModel):
    id: str
    name: str
    class_name: str = Field(default="safe", alias="class")
    criticality: int = 1  # 1 to 3
    mode: str = "fixed"   # fixed or mobile
    camera_id: str = "cam-1"
    tag_id: Optional[int] = None
    region: List[List[int]] = Field(default_factory=list) # [[x1,y1], [x2,y2], ...]
    zone: List[List[int]] = Field(default_factory=list)
    thresholds: AssetThresholds = Field(default_factory=AssetThresholds)


class CameraConfig(BaseModel):
    id: str = "cam-1"
    name: str = "Branch Camera"
    source: str = "synthetic" # synthetic, webcam:0, file:..., rtsp://...
    process_fps: int = 5
    width: int = 1280
    height: int = 720


class BusinessHoursConfig(BaseModel):
    days: List[str] = Field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    start: str = "09:30"
    end: str = "18:30"


class TimerSettings(BaseModel):
    missing_wait_s: float = 15.0
    new_object_s: float = 8.0
    swap_s: float = 10.0
    obstruction_hold_s: float = 1.5
    obstruction_release_s: float = 3.0
    moved_votes: int = 5
    returned_hold_s: float = 5.0


class TelegramAlertConfig(BaseModel):
    enabled: bool = False
    chat_id: str = ""
    bot_token: str = ""


class NtfyAlertConfig(BaseModel):
    enabled: bool = False
    topic: str = "trace-alerts"


class EscalateConfig(BaseModel):
    high: int = 300
    medium: int = 900


class AlertConfig(BaseModel):
    telegram: TelegramAlertConfig = Field(default_factory=TelegramAlertConfig)
    ntfy: NtfyAlertConfig = Field(default_factory=NtfyAlertConfig)
    escalate_after_s: EscalateConfig = Field(default_factory=EscalateConfig)


class LedgerConfig(BaseModel):
    anchor_every_s: int = 60
    anchor_channels: List[str] = Field(default_factory=lambda: ["local_file"])
    key_file: str = "data/signing_key.hex"


class PrivacyConfig(BaseModel):
    mask_people: bool = True
    retention_days: int = 30


class AppConfig(BaseModel):
    site: str = "demo-branch-01"
    timezone: str = "Asia/Kolkata"
    profile: str = "demo"
    business_hours: BusinessHoursConfig = Field(default_factory=BusinessHoursConfig)
    cameras: List[CameraConfig] = Field(default_factory=list)
    assets: List[AssetConfig] = Field(default_factory=list)
    half_lives_s: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    timers: Dict[str, TimerSettings] = Field(default_factory=dict)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    def get_timers(self) -> TimerSettings:
        profile_key = self.profile if self.profile in self.timers else "demo"
        return self.timers.get(profile_key, TimerSettings())

    def get_half_life(self, asset_class: str) -> float:
        profile_key = self.profile if self.profile in self.half_lives_s else "demo"
        class_table = self.half_lives_s.get(profile_key, {})
        return class_table.get(asset_class, class_table.get("other", 25.0))


def load_config(config_path: str = "config/demo.yaml") -> AppConfig:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw_text = p.read_text(encoding="utf-8")
    expanded = expand_env_vars(raw_text)
    data = yaml.safe_load(expanded)
    return AppConfig.model_validate(data)
