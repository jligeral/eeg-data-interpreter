from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class Hypothesis:
    name: str
    description: str
    confidence: float           # 0.0 – 1.0
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)


@dataclass
class ReasoningReport:
    hypotheses: list[Hypothesis] = field(default_factory=list)
    most_likely: str = ""
    clinical_flags: list[str] = field(default_factory=list)
    recommended_next_steps: list[str] = field(default_factory=list)
    what_would_change_mind: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    embedding_stats: dict = field(default_factory=dict)
    raw_response: str = ""
    model: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "most_likely": self.most_likely,
            "hypotheses": [
                {
                    "name": h.name,
                    "description": h.description,
                    "confidence": h.confidence,
                    "evidence_for": h.evidence_for,
                    "evidence_against": h.evidence_against,
                }
                for h in self.hypotheses
            ],
            "clinical_flags": self.clinical_flags,
            "recommended_next_steps": self.recommended_next_steps,
            "what_would_change_mind": self.what_would_change_mind,
            "caveats": self.caveats,
            "embedding_stats": self.embedding_stats,
            "model": self.model,
            "error": self.error,
        }

    def summarize(self) -> str:
        lines = ["Reasoning Report", "══════════════════════════════════════"]

        if self.error:
            lines.append(f"  ERROR: {self.error}")
            return "\n".join(lines)

        if self.most_likely:
            lines.append(f"  Most likely  : {self.most_likely}")

        if self.hypotheses:
            lines.append("  Hypotheses:")
            for h in sorted(self.hypotheses, key=lambda x: -x.confidence):
                lines.append(f"    [{h.confidence:.0%}] {h.name}: {h.description}")
                for e in h.evidence_for:
                    lines.append(f"          + {e}")
                for e in h.evidence_against:
                    lines.append(f"          - {e}")

        if self.clinical_flags:
            lines.append("  Clinical flags:")
            for f_ in self.clinical_flags:
                lines.append(f"    • {f_}")

        if self.recommended_next_steps:
            lines.append("  Recommended next steps:")
            for s in self.recommended_next_steps:
                lines.append(f"    → {s}")

        if self.what_would_change_mind:
            lines.append("  What would change this assessment:")
            for w in self.what_would_change_mind:
                lines.append(f"    ~ {w}")

        if self.caveats:
            lines.append("  Caveats:")
            for c in self.caveats:
                lines.append(f"    ! {c}")

        if self.embedding_stats:
            lines.append("  LEAD embedding stats:")
            for k, v in self.embedding_stats.items():
                lines.append(f"    {k}: {v}")

        return "\n".join(lines)
