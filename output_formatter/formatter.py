"""
output_formatter/formatter.py — assemble EEGReport from FeatureRecord + ReasoningReport.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

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
    """Extract key AD biomarkers from the FeatureRecord into typed BiomarkerValue objects."""
    markers = []

    # Posterior alpha peak frequency
    apf = feat.alpha_peak_frequency
    markers.append(BiomarkerValue(
        name="Posterior alpha peak frequency",
        value=apf,
        unit="Hz",
        abnormal=apf is not None and apf < 8.0,
        reference_range="8.0–13.0 Hz",
        note="Slowing below 8 Hz is associated with cortical dysfunction in AD." if apf is not None and apf < 8.0 else "",
    ))

    # Theta/alpha ratio
    tar = feat.theta_alpha_ratio
    markers.append(BiomarkerValue(
        name="Theta/alpha power ratio (posterior)",
        value=tar,
        unit="ratio",
        abnormal=tar is not None and tar > 1.0,
        reference_range="< 1.0",
        note="Ratio > 1.0 indicates relative theta increase, consistent with AD-pattern slowing." if tar is not None and tar > 1.0 else "",
    ))

    # Posterior alpha relative power
    pap = feat.posterior_alpha_power
    markers.append(BiomarkerValue(
        name="Posterior alpha relative power",
        value=pap,
        unit="relative",
        abnormal=pap is not None and pap < 0.20,
        reference_range="> 0.20",
        note="Reduced posterior alpha power can reflect disrupted thalamocortical rhythms." if pap is not None and pap < 0.20 else "",
    ))

    # Posterior alpha coherence
    pac = feat.posterior_alpha_coherence
    markers.append(BiomarkerValue(
        name="Posterior alpha coherence",
        value=pac,
        unit="coherence",
        abnormal=pac is not None and pac < 0.50,
        reference_range="> 0.50",
        note="Reduced coherence between posterior regions suggests disrupted long-range connectivity." if pac is not None and pac < 0.50 else "",
    ))

    # AD probability (fine-tuned LEAD)
    adp = feat.lead_ad_probability
    if adp is not None:
        markers.append(BiomarkerValue(
            name="LEAD AD probability (fine-tuned classifier)",
            value=adp,
            unit="probability",
            abnormal=adp > 0.60,
            reference_range="< 0.60",
            note="LEAD model probability of AD-like EEG pattern." if adp > 0.60 else "",
        ))

    # Time-domain complexity biomarkers
    se = feat.sample_entropy
    markers.append(BiomarkerValue(
        name="Sample entropy (mean, m=2, r=0.2σ)",
        value=se,
        unit="nats",
        abnormal=se is not None and se < 0.90,
        reference_range="> 0.90 (healthy adults)",
        note="Reduced complexity consistent with more predictable, less irregular EEG — seen in AD." if se is not None and se < 0.90 else "",
    ))

    hc = feat.hjorth_complexity
    markers.append(BiomarkerValue(
        name="Hjorth complexity (mean)",
        value=hc,
        unit="ratio",
        abnormal=hc is not None and hc < 1.40,
        reference_range="> 1.40",
        note="Reduced waveform complexity; lower values indicate simplified oscillatory dynamics." if hc is not None and hc < 1.40 else "",
    ))

    hm = feat.hjorth_mobility
    markers.append(BiomarkerValue(
        name="Hjorth mobility (mean)",
        value=hm,
        unit="rad/s (normalised)",
        abnormal=False,
        reference_range="N/A (context-dependent)",
        note="",
    ))

    return markers


def _build_uncertainty(
    feat: FeatureRecord,
    reasoning: ReasoningReport,
) -> UncertaintyProfile:
    top_conf = max((h.confidence for h in reasoning.hypotheses), default=0.0)

    # Data quality impact description
    quality_impact = {
        "good": "Recording quality is good; measurements are reliable.",
        "moderate": "Moderate recording quality — some artifacts present. Findings should be interpreted with caution.",
        "poor": "Poor recording quality significantly limits reliability. Clinical correlation is essential.",
        "unknown": "Recording quality could not be assessed.",
    }.get(feat.data_quality, feat.data_quality)

    # Build limiting factors from multiple sources
    limiting = []

    if feat.data_quality in ("moderate", "poor"):
        limiting.append(f"Data quality is {feat.data_quality}; artifacts may distort spectral features.")
    if feat.confounds:
        limiting.append(f"Artifact flags present: {', '.join(feat.confounds)}.")
    if not feat.lead_embeddings_available:
        limiting.append("LEAD embeddings unavailable — deep model features could not be computed.")
    if feat.lead_ad_probability is None:
        limiting.append("AD probability score unavailable — requires a fine-tuned LEAD classifier checkpoint.")
    if len(reasoning.hypotheses) > 1:
        confs = sorted([h.confidence for h in reasoning.hypotheses], reverse=True)
        if len(confs) >= 2 and confs[0] - confs[1] < 0.20:
            limiting.append(
                f"Top two hypotheses have similar confidence ({confs[0]:.0%} vs {confs[1]:.0%}); "
                "findings are not strongly discriminating."
            )
    if not reasoning.hypotheses:
        limiting.append("No hypotheses returned by the reasoning agent.")

    # Caveats: merge reasoning caveats with any pipeline-level caveats
    caveats = list(reasoning.caveats)
    if feat.lead_ad_probability is None and feat.lead_embeddings_available:
        caveats.append(
            "LEAD P-Base checkpoint is a pretrain model; AD probability requires a supervised fine-tune."
        )

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

    Parameters
    ----------
    feat      : FeatureRecord from feature_extractor.extract()
    reasoning : ReasoningReport from reasoning_agent.analyze()

    Returns
    -------
    EEGReport — typed, machine-readable and human-readable.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Objective findings ──────────────────────────────────────────────────
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
        lead_embeddings_available=feat.lead_embeddings_available,
        n_lead_windows=feat.n_windows,
        lead_ad_probability=feat.lead_ad_probability,
        lead_embedding_stats=reasoning.embedding_stats,
        preprocessor_notes=list(feat.preprocessor_notes),
        psd_freqs=list(feat.psd_freqs),
        psd_by_region=dict(feat.psd_by_region),
        lead_ad_per_window=list(feat.lead_ad_per_window),
    )

    # ── Differential diagnosis ──────────────────────────────────────────────
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

    # ── Uncertainty ─────────────────────────────────────────────────────────
    uncertainty = _build_uncertainty(feat, reasoning)

    # ── Next steps ──────────────────────────────────────────────────────────
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
