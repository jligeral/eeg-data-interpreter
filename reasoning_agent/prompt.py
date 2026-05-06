"""
Build the clinical reasoning prompt from a FeatureRecord + model stats.
"""
from __future__ import annotations
import json
from feature_extractor.feature_record import FeatureRecord


_SYSTEM_PROMPT = """\
You are a clinical neurophysiology expert specialising in quantitative EEG (qEEG) \
and EEG-based neurological pattern assessment. You reason carefully, acknowledge uncertainty, \
and always flag when findings could be artifactual or require clinical correlation.

Your task: given quantitative EEG features and LaBraM model outputs extracted from a patient recording, \
produce a structured differential assessment. You must output ONLY valid JSON — \
no markdown fences, no commentary outside the JSON object.
"""

_RESPONSE_SCHEMA = {
    "most_likely": "<string: one-sentence clinical impression>",
    "hypotheses": [
        {
            "name": "<string: e.g. 'Alzheimer-pattern EEG features'>",
            "description": "<string: 1–2 sentence explanation>",
            "confidence": "<float 0.0–1.0>",
            "evidence_for": ["<string>"],
            "evidence_against": ["<string>"],
        }
    ],
    "clinical_flags": ["<string: abnormal findings worth flagging>"],
    "recommended_next_steps": ["<string>"],
    "what_would_change_mind": ["<string>"],
    "caveats": ["<string: data quality issues, artifact concerns, limitations>"],
}


def _fmt_band_power(band_power: dict) -> str:
    if not band_power:
        return "  (not available)"
    lines = []
    for band, regions in band_power.items():
        region_str = ", ".join(f"{r}={v:.3f}" for r, v in regions.items())
        lines.append(f"  {band:<8}: {region_str}")
    return "\n".join(lines)


def _fmt_coherence_pairs(pairs: dict, max_pairs: int = 10) -> str:
    if not pairs:
        return "  (not available)"
    items = sorted(pairs.items(), key=lambda x: x[0])[:max_pairs]
    return "\n".join(f"  {k}: {v:.3f}" for k, v in items)


def _fmt_labram_stats(feat: FeatureRecord, stats: dict) -> str:
    lines = []

    if getattr(feat, "labram_available", False):
        lines.append(f"  Windows processed        : {getattr(feat, 'n_windows', 'N/A')}")
        lines.append(f"  Predicted class          : {getattr(feat, 'labram_prediction', 'N/A')}")
        lines.append(f"  Healthy probability      : {getattr(feat, 'labram_prob_healthy', None)}")
        lines.append(f"  Alzheimer probability    : {getattr(feat, 'labram_prob_alzheimers', None)}")
        lines.append(f"  Other neuro probability  : {getattr(feat, 'labram_prob_other', None)}")
    else:
        lines.append("  LaBraM output unavailable.")

    if stats:
        if stats.get("inter_window_variance") is not None:
            lines.append(f"  Inter-window variance    : {stats.get('inter_window_variance')}")
        if stats.get("embedding_norm") is not None:
            lines.append(f"  Embedding norm           : {stats.get('embedding_norm')}")
        if stats.get("pca_top1_variance_ratio") is not None:
            lines.append(f"  PCA top-1 variance       : {stats.get('pca_top1_variance_ratio'):.3f}")

    return "\n".join(lines)


def build_prompt(feat: FeatureRecord, embedding_stats: dict) -> tuple[str, str]:
    """
    Returns (system_prompt, user_message).
    """
    quality_note = {
        "good": "Recording quality: GOOD — features are reliable.",
        "moderate": "Recording quality: MODERATE — some artifacts present, interpret with caution.",
        "poor": "Recording quality: POOR — significant artifacts or data issues; findings unreliable.",
        "unknown": "Recording quality: UNKNOWN.",
    }.get(feat.data_quality, f"Recording quality: {feat.data_quality}.")

    confound_note = ""
    if feat.confounds:
        confound_note = f"\nArtifact/confound flags: {', '.join(feat.confounds)}"

    alpha_note = (
        f"{feat.alpha_peak_frequency:.2f} Hz"
        + (" ← below 8 Hz" if feat.alpha_peak_frequency < 8.0 else " (8–13 Hz range)")
        if feat.alpha_peak_frequency is not None else "not available"
    )
    tar_note = (
        f"{feat.theta_alpha_ratio:.3f}"
        + (" ← elevated relative to threshold" if feat.theta_alpha_ratio > 1.0 else " (below threshold)")
        if feat.theta_alpha_ratio is not None else "not available"
    )
    coh_note = (
        f"{feat.posterior_alpha_coherence:.3f}"
        + (" ← below threshold" if feat.posterior_alpha_coherence < 0.5 else " (within threshold)")
        if feat.posterior_alpha_coherence is not None else "not available"
    )
    post_alpha_note = (
        f"{feat.posterior_alpha_power:.3f} (relative power)"
        if feat.posterior_alpha_power is not None else "not available"
    )

    labram_note = ""
    if getattr(feat, "labram_available", False):
        labram_note = (
            "\nLaBraM model output:"
            f"\n  Predicted class          : {feat.labram_prediction}"
            f"\n  Healthy probability      : {feat.labram_prob_healthy:.3f}"
            f"\n  Alzheimer probability    : {feat.labram_prob_alzheimers:.3f}"
            f"\n  Other neuro probability  : {feat.labram_prob_other:.3f}"
        )
    else:
        labram_note = "\nLaBraM model output: not available"

    user_message = f"""\
EEG CLINICAL ANALYSIS REQUEST
==============================

{quality_note}{confound_note}
File: {feat.filepath or "unknown"}
Channels: {len(feat.channel_labels)} ({', '.join(feat.channel_labels[:8])}{'...' if len(feat.channel_labels) > 8 else ''})
Sampling rate: {feat.sampling_rate:.0f} Hz

KEY EEG FEATURES
----------------
Posterior alpha peak frequency : {alpha_note}
Theta/alpha power ratio        : {tar_note}
Posterior alpha coherence      : {coh_note}
Posterior alpha relative power : {post_alpha_note}{labram_note}

RELATIVE BAND POWER (by brain region)
--------------------------------------
{_fmt_band_power(feat.band_power_relative)}

POSTERIOR COHERENCE PAIRS (alpha band)
---------------------------------------
{_fmt_coherence_pairs(feat.coherence_pairs)}

LaBraM FOUNDATION MODEL ANALYSIS
--------------------------------
LaBraM is a transformer-based EEG foundation model used here as a three-class classifier:
Healthy, Alzheimer, and Other neurological.
{_fmt_labram_stats(feat, embedding_stats)}

PREPROCESSOR NOTES
------------------
{chr(10).join('  • ' + n for n in feat.preprocessor_notes) or '  (none)'}

REQUESTED OUTPUT
----------------
Provide a structured differential assessment as JSON matching this schema exactly:
{json.dumps(_RESPONSE_SCHEMA, indent=2)}

Requirements:
- Consider the full pattern of findings, not individual metrics in isolation.
- Use LaBraM probabilities as model evidence, not as a standalone diagnosis.
- Compare the biomarker pattern against the LaBraM output.
- Be calibrated: high confidence (>0.7) only if multiple independent findings align.
- Flag data quality concerns in caveats.
- recommended_next_steps should be clinically practical.
- Do not diagnose — frame as probability/pattern assessment.
"""
    return _SYSTEM_PROMPT, user_message