"""Plain data containers shared by all modules (no heavy imports here)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualityReport:
    ok: bool
    issues: list = field(default_factory=list)   # codes: too_small, blurry, too_dark, overexposed
    metrics: dict = field(default_factory=dict)


@dataclass
class GateResult:
    verdict: str          # "ok" | "not_plant" | "leaf_not_visible"
    plant_prob: float
    leaf_prob: float


@dataclass
class SymptomCue:
    key: str              # e.g. "yellowing"
    score: float          # zero-shot score 0-1 (indicative only)


@dataclass
class Candidate:
    label: str
    species: str
    condition: str
    prob: float


@dataclass
class Readings:
    """Optional user/sensor-supplied measurements. None = not supplied."""
    soil_moisture_pct: Optional[float] = None
    air_temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    light_hours: Optional[float] = None
    source: str = "manual"
