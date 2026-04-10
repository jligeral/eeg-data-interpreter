"""
output_formatter/eeg_report.py — typed EEGReport and its four section dataclasses.

EEGReport is the final deliverable of the pipeline. It is:
  - Machine-readable via to_dict() / to_json()
  - Human-readable via to_text()

Sections
--------
  ObjectiveFindings      — measured values, no interpretation
  DifferentialDiagnosis  — ranked hypotheses from the reasoning agent
  UncertaintyProfile     — overall confidence, data quality, limiting factors
  NextSteps              — recommended clinical and research actions
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Section 1: Objective Findings
# ---------------------------------------------------------------------------

@dataclass
class BiomarkerValue:
    """A single measured biomarker with its value, unit, and normal-range flag."""
    name: str
    value: Optional[float]
    unit: str
    abnormal: bool              # True if outside the reference range for this pipeline
    reference_range: str        # e.g. "8.0–13.0 Hz"
    note: str = ""              # one-line plain-language description


@dataclass
class ObjectiveFindings:
    """
    Raw measured values from the EEG record.  No interpretation — just what
    the instruments and algorithms measured.
    """
    source_file: str
    file_format: str
    data_quality: str           # "good" | "moderate" | "poor" | "unknown"
    n_channels: int
    channel_labels: list[str]
    sampling_rate_hz: float
    confounds: list[str]        # artifact flags from preprocessor

    # Key AD biomarkers
    biomarkers: list[BiomarkerValue]

    # Band power (relative) by region — {"delta": {"frontal": 0.06, ...}, ...}
    band_power_relative: dict
    band_power_absolute: dict

    # LEAD foundation model
    lead_embeddings_available: bool
    n_lead_windows: int
    lead_ad_probability: Optional[float]    # None unless fine-tuned classifier present
    lead_embedding_stats: dict              # inter-window variance, norm, PCA ratio, etc.

    preprocessor_notes: list[str]


# ---------------------------------------------------------------------------
# Section 2: Differential Diagnosis
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticHypothesis:
    """One ranked hypothesis in the differential."""
    rank: int
    name: str
    description: str
    confidence: float           # 0.0 – 1.0
    confidence_label: str       # "high" | "moderate" | "low"
    supporting_evidence: list[str]
    contradicting_evidence: list[str]


@dataclass
class DifferentialDiagnosis:
    clinical_impression: str    # one-sentence overall impression
    hypotheses: list[DiagnosticHypothesis]
    clinical_flags: list[str]   # abnormal findings worth flagging directly


# ---------------------------------------------------------------------------
# Section 3: Uncertainty Profile
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyProfile:
    """
    Explicit characterisation of confidence and what limits it.
    Separates data-quality uncertainty (fixable with better recording) from
    model uncertainty (requires more evidence regardless of data quality).
    """
    overall_confidence: float       # 0.0 – 1.0, from top hypothesis
    overall_confidence_label: str   # "high" | "moderate" | "low"
    data_quality_impact: str        # plain-language description
    limiting_factors: list[str]     # specific reasons confidence is not higher
    caveats: list[str]              # model/method limitations
    what_would_change_assessment: list[str]


# ---------------------------------------------------------------------------
# Section 4: Next Steps
# ---------------------------------------------------------------------------

@dataclass
class NextSteps:
    recommended_actions: list[str]


# ---------------------------------------------------------------------------
# Assembled report
# ---------------------------------------------------------------------------

@dataclass
class EEGReport:
    generated_at: str                   # ISO-8601 UTC timestamp
    pipeline_model: str                 # reasoning model name (e.g. qwen3:8b)
    reasoning_error: Optional[str]      # non-None if reasoning agent failed

    objective_findings: ObjectiveFindings
    differential_diagnosis: DifferentialDiagnosis
    uncertainty: UncertaintyProfile
    next_steps: NextSteps

    # -----------------------------------------------------------------------
    # Machine-readable output
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        f = self.objective_findings
        dd = self.differential_diagnosis
        u = self.uncertainty
        ns = self.next_steps

        return {
            "generated_at": self.generated_at,
            "pipeline_model": self.pipeline_model,
            "reasoning_error": self.reasoning_error,
            "objective_findings": {
                "source_file": f.source_file,
                "file_format": f.file_format,
                "data_quality": f.data_quality,
                "n_channels": f.n_channels,
                "channel_labels": f.channel_labels,
                "sampling_rate_hz": f.sampling_rate_hz,
                "confounds": f.confounds,
                "biomarkers": [
                    {
                        "name": b.name,
                        "value": b.value,
                        "unit": b.unit,
                        "abnormal": b.abnormal,
                        "reference_range": b.reference_range,
                        "note": b.note,
                    }
                    for b in f.biomarkers
                ],
                "band_power_relative": f.band_power_relative,
                "band_power_absolute": f.band_power_absolute,
                "lead_embeddings_available": f.lead_embeddings_available,
                "n_lead_windows": f.n_lead_windows,
                "lead_ad_probability": f.lead_ad_probability,
                "lead_embedding_stats": f.lead_embedding_stats,
                "preprocessor_notes": f.preprocessor_notes,
            },
            "differential_diagnosis": {
                "clinical_impression": dd.clinical_impression,
                "clinical_flags": dd.clinical_flags,
                "hypotheses": [
                    {
                        "rank": h.rank,
                        "name": h.name,
                        "description": h.description,
                        "confidence": h.confidence,
                        "confidence_label": h.confidence_label,
                        "supporting_evidence": h.supporting_evidence,
                        "contradicting_evidence": h.contradicting_evidence,
                    }
                    for h in dd.hypotheses
                ],
            },
            "uncertainty": {
                "overall_confidence": u.overall_confidence,
                "overall_confidence_label": u.overall_confidence_label,
                "data_quality_impact": u.data_quality_impact,
                "limiting_factors": u.limiting_factors,
                "caveats": u.caveats,
                "what_would_change_assessment": u.what_would_change_assessment,
            },
            "next_steps": {
                "recommended_actions": ns.recommended_actions,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # -----------------------------------------------------------------------
    # Human-readable output
    # -----------------------------------------------------------------------

    def to_text(self) -> str:
        lines: list[str] = []

        def rule(char: str = "─", width: int = 60) -> str:
            return char * width

        def section(title: str) -> None:
            lines.append("")
            lines.append(rule("═"))
            lines.append(f"  {title.upper()}")
            lines.append(rule("═"))

        def sub(title: str) -> None:
            lines.append(f"\n  {title}")
            lines.append("  " + rule("─", 40))

        # Header
        lines.append(rule("═"))
        lines.append("  EEG CLINICAL ANALYSIS REPORT")
        lines.append(f"  Generated : {self.generated_at}")
        lines.append(f"  Model     : {self.pipeline_model}")
        lines.append(rule("═"))

        if self.reasoning_error:
            lines.append(f"\n  [!] REASONING ERROR: {self.reasoning_error}")
            lines.append("  Objective findings below are still valid.\n")

        # ── Section 1: Objective Findings ──────────────────────────────────
        section("1. Objective Findings")
        f = self.objective_findings

        quality_sym = {"good": "[GOOD]", "moderate": "[MODERATE]", "poor": "[POOR]"}.get(
            f.data_quality, f"[{f.data_quality.upper()}]"
        )
        lines.append(f"\n  File    : {f.source_file or '(not specified)'}")
        lines.append(f"  Format  : {f.file_format}")
        lines.append(f"  Quality : {quality_sym}")
        lines.append(f"  Channels: {f.n_channels}  |  Sampling rate: {f.sampling_rate_hz:.0f} Hz")
        if f.confounds:
            lines.append(f"  Artifact flags: {', '.join(f.confounds)}")

        sub("Key AD Biomarkers")
        for b in f.biomarkers:
            flag = "  <-- ABNORMAL" if b.abnormal else ""
            val_str = f"{b.value:.3f} {b.unit}" if b.value is not None else "N/A"
            lines.append(f"  {b.name:<38} {val_str:<18} (ref: {b.reference_range}){flag}")
            if b.note:
                lines.append(f"    {b.note}")

        if f.lead_embeddings_available:
            sub("LEAD Foundation Model")
            lines.append(f"  Windows encoded : {f.n_lead_windows}")
            if f.lead_ad_probability is not None:
                ad_flag = "  <-- ELEVATED" if f.lead_ad_probability > 0.6 else ""
                lines.append(f"  AD probability  : {f.lead_ad_probability:.3f}{ad_flag}")
            else:
                lines.append("  AD probability  : N/A (pretrain checkpoint only)")
            if f.lead_embedding_stats:
                es = f.lead_embedding_stats
                lines.append(f"  Inter-win var   : {es.get('inter_window_variance', 'N/A')}")
                lines.append(f"  Embedding norm  : {es.get('embedding_norm', 'N/A')}")
                pca = es.get('pca_top1_variance_ratio')
                if pca is not None:
                    lines.append(f"  PCA top-1 var   : {pca:.3f}")
                top = es.get('highest_activation_channels', [])
                if top:
                    lines.append(f"  Peak channels   : {', '.join(top)}")

        if f.band_power_relative:
            sub("Relative Band Power (by region)")
            for band, regions in f.band_power_relative.items():
                region_str = "  ".join(f"{r}={v:.3f}" for r, v in regions.items())
                lines.append(f"  {band:<8}: {region_str}")

        # ── Section 2: Differential Diagnosis ──────────────────────────────
        section("2. Differential Diagnosis")
        dd = self.differential_diagnosis

        if dd.clinical_impression:
            lines.append(f"\n  Impression: {dd.clinical_impression}\n")

        if dd.clinical_flags:
            lines.append("  Clinical flags:")
            for flag in dd.clinical_flags:
                lines.append(f"    [!] {flag}")
            lines.append("")

        for h in dd.hypotheses:
            conf_bar = _confidence_bar(h.confidence)
            lines.append(f"  #{h.rank}  {h.name}  [{h.confidence:.0%} — {h.confidence_label.upper()}]")
            lines.append(f"      {conf_bar}")
            lines.append(f"      {h.description}")
            if h.supporting_evidence:
                lines.append("      Supporting:")
                for e in h.supporting_evidence:
                    lines.append(f"        + {e}")
            if h.contradicting_evidence:
                lines.append("      Against:")
                for e in h.contradicting_evidence:
                    lines.append(f"        - {e}")
            lines.append("")

        # ── Section 3: Uncertainty & Confidence ────────────────────────────
        section("3. Uncertainty & Confidence")
        u = self.uncertainty

        lines.append(f"\n  Overall confidence : {u.overall_confidence:.0%}  [{u.overall_confidence_label.upper()}]")
        lines.append(f"  Data quality impact: {u.data_quality_impact}")

        if u.limiting_factors:
            lines.append("\n  Limiting factors:")
            for lf in u.limiting_factors:
                lines.append(f"    • {lf}")

        if u.caveats:
            lines.append("\n  Caveats:")
            for c in u.caveats:
                lines.append(f"    ! {c}")

        if u.what_would_change_assessment:
            lines.append("\n  What would change this assessment:")
            for w in u.what_would_change_assessment:
                lines.append(f"    ~ {w}")

        # ── Section 4: Next Steps ───────────────────────────────────────────
        section("4. Next Steps")
        lines.append("")
        for i, action in enumerate(self.next_steps.recommended_actions, 1):
            lines.append(f"  {i}. {action}")

        lines.append("")
        lines.append(rule("═"))

        return "\n".join(lines)


def _confidence_bar(confidence: float, width: int = 20) -> str:
    filled = round(confidence * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"
