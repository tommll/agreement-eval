"""Experiment configuration (YAML -> dataclasses)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .extract.prompts import get_prompt
from .schemas import get_schema


@dataclass
class ExtractionConfig:
    """One model/prompt configuration - a 'rater' in the agreement analysis."""

    id: str
    provider: str = "anthropic"
    model: str = "claude-opus-5"
    prompt: str = "detailed"
    input_mode: str = "text"          # text | image | both
    effort: Optional[str] = "medium"  # output_config.effort; None omits it
    max_tokens: int = 4000
    max_ocr_chars: int = 8000
    params: Dict[str, Any] = field(default_factory=dict)  # provider-specific knobs

    def __post_init__(self) -> None:
        if self.input_mode not in ("text", "image", "both"):
            raise ValueError(f"{self.id}: input_mode must be text|image|both, got {self.input_mode!r}")
        get_prompt(self.prompt)  # fail fast on a typo'd variant

    def to_row(self, schema_name: str) -> Dict[str, Any]:
        prompt = get_prompt(self.prompt)
        return {
            "config_id": self.id,
            "provider": self.provider,
            "model": self.model,
            "prompt_variant": self.prompt,
            "input_mode": self.input_mode,
            "effort": self.effort,
            "prompt_version": prompt.version(get_schema(schema_name)),
            "params": {"max_tokens": self.max_tokens, "max_ocr_chars": self.max_ocr_chars, **self.params},
        }


@dataclass
class JudgeConfig:
    """LLM judge used for semantic agreement and semantic correctness."""

    enabled: bool = True
    provider: str = "anthropic"
    model: str = "claude-opus-5"
    effort: Optional[str] = "low"
    max_tokens: int = 4000
    batch_size: int = 20
    # Judge only the pairs that exact match already rejects: identical strings
    # need no judge, and skipping them is most of the cost.
    only_disagreements: bool = True
    judge_gold: bool = True   # also judge prediction-vs-gold (semantic accuracy)
    # Free-text rationales are the audit trail, and also where a small local
    # model degenerates. Keep them for a frontier judge, drop them for a local one.
    include_rationale: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    run_id: str
    dataset: str
    schema: Optional[str] = None
    split: Optional[str] = None
    limit: Optional[int] = None
    concurrency: int = 4
    notes: Optional[str] = None
    ingest: Dict[str, Any] = field(default_factory=dict)
    configs: List[ExtractionConfig] = field(default_factory=list)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def __post_init__(self) -> None:
        if not self.configs:
            raise ValueError("experiment needs at least one extraction config")
        ids = [c.id for c in self.configs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate config ids: {ids}")
        if len(ids) < 3:
            # Not fatal: a 2-config run still produces pairwise agreement, but
            # Dawid-Skene and majority vote need >= 3 to be meaningful.
            print(f"warning: only {len(ids)} configs - majority vote and Dawid-Skene want 3+")
        get_schema(self.schema_name)

    @property
    def schema_name(self) -> str:
        return self.schema or self.dataset

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_experiment(path: str | Path) -> Experiment:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    configs = [ExtractionConfig(**c) for c in raw.pop("configs", [])]
    judge = JudgeConfig(**(raw.pop("judge", {}) or {}))
    return Experiment(configs=configs, judge=judge, **raw)
