"""Central, validated configuration. Every threshold used by an analysis lives here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int = 42
    # Ingestion
    max_upload_mb: int = 50
    max_zip_uncompressed_mb: int = 200
    # Behaviour analytics
    session_gap_minutes: int = 60          # silence that starts a new conversation session
    response_window_minutes: int = 360     # replies later than this are not counted as responses
    # Text analytics
    min_ngram_count: int = 3
    top_n: int = 20
    # Parsing
    extra_media_markers: tuple[str, ...] = field(default_factory=tuple)
    # Statistics
    alpha: float = 0.05
    bootstrap_iterations: int = 2000

    def __post_init__(self) -> None:
        if self.session_gap_minutes <= 0 or self.response_window_minutes <= 0:
            raise ValueError("gap/window minutes must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnalysisConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        if "extra_media_markers" in data:
            data["extra_media_markers"] = tuple(data["extra_media_markers"])
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
