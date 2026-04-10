"""
Build the clinical reasoning prompt from a FeatureRecord + embedding stats.
"""
from __future__ import annotations
import json
from feature_extractor.feature_record import FeatureRecord


_SYSTEM_PROMPT = """\
You are a clinical neurophysiology expert specialising in quantitative EEG (qEEG) \
and Alzheimer's disease biomarkers. You reason carefully, acknowledge uncertainty, \
and always flag when findings could be artifactual or require clinical correlation.

Your task: given quantitative EEG features extracted from a patient recording, \
produce a structured differential assessment. You must output ONLY valid JSON — \
no markdown fences, no commentary outside the JSON object.
"""

_RESPONSE_SCHEMA = {
    "most_likely": "<string: one-sentence clinical impression>",
    "hypotheses": [
        {
            "name": "<string: e.g. 'Early AD pattern'>",
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


def _fmt_embedding_stats(stats: dict) -> str:
    if not stats:
        return "  (LEAD embeddings unavailable)"
    lines = [
        f"  Windows processed     : {stats.get('n_windows', 'N/A')}",
        f"  Inter-window variance : {stats.get('inter_window_variance', 'N/A')} "
        f"(higher = more dynamic instability)",
        f"  Embedding norm        : {stats.get('embedding_norm', 'N/A')}",
    ]
    pca = stats.get("pca_top1_variance_ratio")
    if pca is not None:
        lines.append(f"  PCA top-1 variance    : {pca:.3f} (lower = more diffuse activation)")
    top = stats.get("highest_activation_channels", [])
    low = stats.get("lowest_activation_channels", [])
    if top:
        lines.append(f"  Highest activation    : {', '.join(top)} (channels driving embedding most)")
    if low:
        lines.append(f"  Lowest activation     : {', '.join(low)}")
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

    # Key AD biomarkers section
    alpha_note = (
        f"{feat.alpha_peak_frequency:.2f} Hz"
        + (" ← SLOWED (< 8 Hz, consistent with AD-related cortical slowing)" if feat.alpha_peak_frequency < 8.0 else " (normal range 8–13 Hz)")
        if feat.alpha_peak_frequency is not None else "not available"
    )
    tar_note = (
        f"{feat.theta_alpha_ratio:.3f}"
        + (" ← ELEVATED (> 1.0, associated with AD-pattern slowing)" if feat.theta_alpha_ratio > 1.0 else " (normal < 1.0)")
        if feat.theta_alpha_ratio is not None else "not available"
    )
    coh_note = (
        f"{feat.posterior_alpha_coherence:.3f}"
        + (" ← REDUCED (< 0.5, consistent with disrupted posterior connectivity)" if feat.posterior_alpha_coherence < 0.5 else " (within range)")
        if feat.posterior_alpha_coherence is not None else "not available"
    )
    post_alpha_note = (
        f"{feat.posterior_alpha_power:.3f} (relative power)"
        if feat.posterior_alpha_power is not None else "not available"
    )

    lead_note = ""
    if feat.lead_ad_probability is not None:
        flag = " ← ELEVATED" if feat.lead_ad_probability > 0.6 else ""
        lead_note = f"\nLEAD AD probability (fine-tuned classifier): {feat.lead_ad_probability:.3f}{flag}"
    else:
        lead_note = "\nLEAD AD probability: not available (P-Base pretrain checkpoint; no trained classifier)"

    user_message = f"""\
EEG CLINICAL ANALYSIS REQUEST
==============================

{quality_note}{confound_note}
File: {feat.filepath or "unknown"}
Channels: {len(feat.channel_labels)} ({', '.join(feat.channel_labels[:8])}{'...' if len(feat.channel_labels) > 8 else ''})
Sampling rate: {feat.sampling_rate:.0f} Hz

KEY AD BIOMARKERS
-----------------
Posterior alpha peak frequency : {alpha_note}
Theta/alpha power ratio        : {tar_note}
Posterior alpha coherence      : {coh_note}
Posterior alpha relative power : {post_alpha_note}{lead_note}

RELATIVE BAND POWER (by brain region)
--------------------------------------
{_fmt_band_power(feat.band_power_relative)}

POSTERIOR COHERENCE PAIRS (alpha band)
---------------------------------------
{_fmt_coherence_pairs(feat.coherence_pairs)}

LEAD FOUNDATION MODEL EMBEDDING ANALYSIS
-----------------------------------------
(LEAD is a transformer pre-trained on 13 EEG datasets for Alzheimer's detection)
{_fmt_embedding_stats(embedding_stats)}

PREPROCESSOR NOTES
------------------
{chr(10).join('  • ' + n for n in feat.preprocessor_notes) or '  (none)'}

REQUESTED OUTPUT
----------------
Provide a structured differential assessment as JSON matching this schema exactly:
{json.dumps(_RESPONSE_SCHEMA, indent=2)}

Requirements:
- Consider the full pattern of findings, not individual metrics in isolation.
- Explicitly note which findings are consistent vs inconsistent with AD.
- Be calibrated: high confidence (>0.7) only if multiple independent markers align.
- Flag data quality concerns in caveats.
- recommended_next_steps should be clinically practical (e.g., cognitive screening, structural MRI, repeat EEG under controlled conditions).
- Do not diagnose — frame as probability/pattern assessment.
"""
    return _SYSTEM_PROMPT, user_message
