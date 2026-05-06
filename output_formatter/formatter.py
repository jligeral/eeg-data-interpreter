"""
output_formatter/formatter.py — assemble EEGReport from FeatureRecord + ReasoningReport.
"""
from __future__ import annotations

from datetime import datetime, timezone

from feature_extractor.feature_record import FeatureRecord
from reasoning_agent.report import ReasoningReport
from output_formatter.eeg_report import (
    BiomarkerValue,
    ObjectiveFindings,
    DiagnosticHypothesis,
    DifferentialDiagnosis,
    UncertaintyProfile,
    NextSteps,
    EEGReport,
)


def _confidence_label(c: float) -> str:
    if c >= 0.70:
        return "high"
    if c >= 0.40:
        return "moderate"
    return "low"


def _build_biomarkers(feat: FeatureRecord) -> list[BiomarkerValue]:
    """Extract key biomarkers and LaBraM outputs from the FeatureRecord."""
    markers = []

    apf = feat.alpha_peak_frequency
    markers.append(BiomarkerValue(
        name="Posterior alpha peak frequency",
        value=apf,
        unit="Hz",
        abnormal=apf is not None and apf < 8.0,
        reference_range="8.0–13.0 Hz",
        note="Below expected alpha range." if apf is not None and apf < 8.0 else "",
    ))

    tar = feat.theta_alpha_ratio
    markers.append(BiomarkerValue(
        name="Theta/alpha power ratio",
        value=tar,
        unit="ratio",
        abnormal=tar is not None and tar > 1.0,
        reference_range="< 1.0",
        note="Theta/alpha ratio is elevated." if tar is not None and tar > 1.0 else "",
    ))

    pap = feat.posterior_alpha_power
    markers.append(BiomarkerValue(
        name="Posterior alpha relative power",
        value=pap,
        unit="relative",
        abnormal=pap is not None and pap < 0.20,
        reference_range="> 0.20",
        note="Posterior alpha relative power is reduced." if pap is not None and pap < 0.20 else "",
    ))

    pac = feat.posterior_alpha_coherence
    markers.append(BiomarkerValue(
        name="Posterior alpha coherence",
        value=pac,
        unit="coherence",
        abnormal=pac is not None and pac < 0.50,
        reference_range="> 0.50",
        note="Posterior alpha coherence is reduced." if pac is not None and pac < 0.50 else "",
    ))

    if getattr(feat, "labram_prob_healthy", None) is not None:
        markers.append(BiomarkerValue(
            name="LaBraM healthy probability",
            value=feat.labram_prob_healthy,
            unit="probability",
            abnormal=False,
            reference_range="0.0–1.0",
            note="Subject-level probability averaged across LaBraM windows.",
        ))

    if getattr(feat, "labram_prob_alzheimers", None) is not None:
        markers.append(BiomarkerValue(
            name="LaBraM Alzheimer probability",
            value=feat.labram_prob_alzheimers,
            unit="probability",
            abnormal=feat.labram_prob_alzheimers > 0.60,
            reference_range="0.0–1.0",
            note="Subject-level probability averaged across LaBraM windows.",
        ))

    if getattr(feat, "labram_prob_other", None) is not None:
        markers.append(BiomarkerValue(
            name="LaBraM other neurological probability",
            value=feat.labram_prob_other,
            unit="probability",
            abnormal=feat.labram_prob_other > 0.60,
            reference_range="0.0–1.0",
            note="Subject-level probability averaged across LaBraM windows.",
        ))

    return markers


def _build_uncertainty(
    feat: FeatureRecord,
    reasoning: ReasoningReport,
) -> UncertaintyProfile:
    top_conf = max((h.confidence for h in reasoning.hypotheses), default=0.0)

    quality_impact = {
        "good": "Recording quality is good; measurements are reliable.",
        "moderate": "Moderate recording quality — some artifacts present. Findings should be interpreted with caution.",
        "poor": "Poor recording quality significantly limits reliability. Clinical correlation is essential.",
        "unknown": "Recording quality could not be assessed.",
    }.get(feat.data_quality, feat.data_quality)

    limiting = []

    if feat.data_quality in ("moderate", "poor"):
        limiting.append(f"Data quality is {feat.data_quality}; artifacts may distort spectral features.")
    if feat.confounds:
        limiting.append(f"Artifact flags present: {', '.join(feat.confounds)}.")

    if not getattr(feat, "labram_available", False):
        limiting.append("LaBraM output unavailable — deep model probabilities could not be computed.")

    if getattr(feat, "labram_prob_alzheimers", None) is None:
        limiting.append("LaBraM Alzheimer probability unavailable.")

    if len(reasoning.hypotheses) > 1:
        confs = sorted([h.confidence for h in reasoning.hypotheses], reverse=True)
        if len(confs) >= 2 and confs[0] - confs[1] < 0.20:
            limiting.append(
                f"Top two hypotheses have similar confidence ({confs[0]:.0%} vs {confs[1]:.0%}); "
                "findings are not strongly discriminating."
            )

    if not reasoning.hypotheses:
        limiting.append("No hypotheses returned by the reasoning agent.")

    caveats = list(reasoning.caveats)

    return UncertaintyProfile(
        overall_confidence=top_conf,
        overall_confidence_label=_confidence_label(top_conf),
        data_quality_impact=quality_impact,
        limiting_factors=limiting,
        caveats=caveats,
        what_would_change_assessment=reasoning.what_would_change_mind,
    )


def format_report(
    feat: FeatureRecord,
    reasoning: ReasoningReport,
) -> EEGReport:
    """
    Assemble a final EEGReport from the feature record and reasoning report.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    objective = ObjectiveFindings(
        source_file=feat.filepath,
        file_format=feat.file_format,
        data_quality=feat.data_quality,
        n_channels=len(feat.channel_labels),
        channel_labels=feat.channel_labels,
        sampling_rate_hz=feat.sampling_rate,
        confounds=list(feat.confounds),
        biomarkers=_build_biomarkers(feat),
        band_power_relative=feat.band_power_relative,
        band_power_absolute=feat.band_power_absolute,

        # Kept using existing EEGReport field names so the rest of the app does not break.
        # These now store LaBraM information instead of LEAD information.
        lead_embeddings_available=getattr(feat, "labram_available", False),
        n_lead_windows=feat.n_windows,
        lead_ad_probability=getattr(feat, "labram_prob_alzheimers", None),
        lead_embedding_stats=reasoning.embedding_stats,

        preprocessor_notes=list(feat.preprocessor_notes) + list(feat.processing_notes),
    )

    hypotheses = [
        DiagnosticHypothesis(
            rank=i + 1,
            name=h.name,
            description=h.description,
            confidence=h.confidence,
            confidence_label=_confidence_label(h.confidence),
            supporting_evidence=list(h.evidence_for),
            contradicting_evidence=list(h.evidence_against),
        )
        for i, h in enumerate(
            sorted(reasoning.hypotheses, key=lambda x: -x.confidence)
        )
    ]

    differential = DifferentialDiagnosis(
        clinical_impression=reasoning.most_likely,
        hypotheses=hypotheses,
        clinical_flags=list(reasoning.clinical_flags),
    )

    uncertainty = _build_uncertainty(feat, reasoning)

    next_steps = NextSteps(
        recommended_actions=list(reasoning.recommended_next_steps),
    )

    return EEGReport(
        generated_at=timestamp,
        pipeline_model=reasoning.model,
        reasoning_error=reasoning.error,
        objective_findings=objective,
        differential_diagnosis=differential,
        uncertainty=uncertainty,
        next_steps=next_steps,
    )