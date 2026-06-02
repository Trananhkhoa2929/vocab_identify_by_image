from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class QualityReport:
    ok: bool
    width: int
    height: int
    blur_score: float
    brightness: float
    contrast: float
    dark_fraction: float
    bright_fraction: float
    warnings: list[str]
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass
class Detection:
    label: str
    bbox: tuple[int,int,int,int]
    detector_conf: float
    detector_label: str | None = None
    final_conf: float | None = None
    verified: bool = False
    verifier_label: str | None = None
    verifier_prob: float | None = None
    verifier_margin: float | None = None
    difficulty: float = 0.0
    source: str = "detector"
    def score(self) -> float: return float(self.final_conf if self.final_conf is not None else self.detector_conf)
    def to_dict(self):
        d=asdict(self); d['bbox']=list(self.bbox); return d

@dataclass
class StudyNote:
    object_label: str
    native_text: str
    target_text: str
    target_ipa: str
    pos: str
    example: str
    confidence: float
    count: int = 1
    def to_dict(self): return asdict(self)
