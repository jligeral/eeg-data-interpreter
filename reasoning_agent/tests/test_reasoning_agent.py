"""
Unit tests for the reasoning agent — Ollama calls are mocked.
"""
import json
import numpy as np
import pytest
from unittest.mock import patch

from feature_extractor.feature_record import FeatureRecord
from feature_extractor.windower import LEAD_CHANNEL_ORDER
from reasoning_agent.embedding_analysis import compute_embedding_stats
from reasoning_agent.prompt import build_prompt
from reasoning_agent.report import ReasoningReport, Hypothesis
from reasoning_agent.agent import analyze

_N_CHANNELS = 19
_D_MODEL = 128
_EMBED_DIM = _N_CHANNELS * _D_MODEL  # 2432


def _make_feat(
    alpha_peak=10.0,
    theta_alpha=0.8,
    coherence=0.6,
    quality="good",
    with_embeddings=True,
    n_windows=10,
):
    feat = FeatureRecord(
        filepath="test.fif",
        file_format=".fif",
        data_quality=quality,
        channel_labels=list(LEAD_CHANNEL_ORDER),
        sampling_rate=256.0,
        alpha_peak_frequency=alpha_peak,
        theta_alpha_ratio=theta_alpha,
        posterior_alpha_power=0.35,
        posterior_alpha_coherence=coherence,
        coherence_pairs={"O1-O2": 0.6, "P3-P4": 0.65},
        band_power_relative={
            "delta": {"frontal": 0.06, "occipital": 0.01},
            "theta": {"frontal": 0.10, "occipital": 0.40},
            "alpha": {"frontal": 0.14, "occipital": 0.48},
            "beta":  {"frontal": 0.31, "occipital": 0.04},
        },
    )
    if with_embeddings:
        rng = np.random.default_rng(0)
        feat.lead_embeddings = rng.standard_normal((n_windows, _EMBED_DIM)).astype(np.float32)
        feat.lead_embeddings_available = True
        feat.n_windows = n_windows
    return feat


# ---------------------------------------------------------------------------
# Embedding analysis
# ---------------------------------------------------------------------------

class TestEmbeddingAnalysis:

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((10, _EMBED_DIM)).astype(np.float32)
        stats = compute_embedding_stats(emb, list(LEAD_CHANNEL_ORDER))
        for key in ("n_windows", "inter_window_variance", "embedding_norm",
                    "pca_top1_variance_ratio", "channel_block_norms",
                    "highest_activation_channels", "lowest_activation_channels"):
            assert key in stats

    def test_channel_block_norms_length(self):
        rng = np.random.default_rng(1)
        emb = rng.standard_normal((5, _EMBED_DIM)).astype(np.float32)
        stats = compute_embedding_stats(emb, list(LEAD_CHANNEL_ORDER))
        assert len(stats["channel_block_norms"]) == _N_CHANNELS

    def test_empty_embeddings_returns_empty(self):
        stats = compute_embedding_stats(None)
        assert stats == {}

    def test_top_low_channels_disjoint(self):
        rng = np.random.default_rng(2)
        emb = rng.standard_normal((8, _EMBED_DIM)).astype(np.float32)
        stats = compute_embedding_stats(emb, list(LEAD_CHANNEL_ORDER))
        top = set(stats["highest_activation_channels"])
        low = set(stats["lowest_activation_channels"])
        assert top.isdisjoint(low)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:

    def test_returns_two_strings(self):
        feat = _make_feat()
        stats = compute_embedding_stats(feat.lead_embeddings, feat.channel_labels)
        sys_p, user_p = build_prompt(feat, stats)
        assert isinstance(sys_p, str) and len(sys_p) > 10
        assert isinstance(user_p, str) and len(user_p) > 10

    def test_slowed_alpha_flagged_in_prompt(self):
        feat = _make_feat(alpha_peak=7.0)
        _, user_p = build_prompt(feat, {})
        assert "SLOWED" in user_p

    def test_elevated_tar_flagged_in_prompt(self):
        feat = _make_feat(theta_alpha=1.5)
        _, user_p = build_prompt(feat, {})
        assert "ELEVATED" in user_p

    def test_poor_quality_noted(self):
        feat = _make_feat(quality="poor")
        _, user_p = build_prompt(feat, {})
        assert "POOR" in user_p


# ---------------------------------------------------------------------------
# Reasoning agent (mocked Ollama)
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = json.dumps({
    "most_likely": "Normal EEG pattern",
    "hypotheses": [
        {
            "name": "Normal",
            "description": "All markers within normal range.",
            "confidence": 0.8,
            "evidence_for": ["Alpha peak at 10 Hz", "Theta/alpha ratio < 1.0"],
            "evidence_against": [],
        }
    ],
    "clinical_flags": [],
    "recommended_next_steps": ["No immediate action required."],
    "what_would_change_mind": ["Elevated theta/alpha ratio on repeat EEG"],
    "caveats": ["Synthetic test data"],
})


class TestAnalyze:

    def test_returns_report(self):
        feat = _make_feat()
        with patch("reasoning_agent.agent._call_ollama", return_value=_MOCK_RESPONSE):
            report = analyze(feat)
        assert isinstance(report, ReasoningReport)

    def test_hypotheses_parsed(self):
        feat = _make_feat()
        with patch("reasoning_agent.agent._call_ollama", return_value=_MOCK_RESPONSE):
            report = analyze(feat)
        assert len(report.hypotheses) == 1
        assert report.hypotheses[0].name == "Normal"
        assert report.hypotheses[0].confidence == 0.8

    def test_most_likely_populated(self):
        feat = _make_feat()
        with patch("reasoning_agent.agent._call_ollama", return_value=_MOCK_RESPONSE):
            report = analyze(feat)
        assert report.most_likely == "Normal EEG pattern"

    def test_embedding_stats_attached(self):
        feat = _make_feat(with_embeddings=True)
        with patch("reasoning_agent.agent._call_ollama", return_value=_MOCK_RESPONSE):
            report = analyze(feat)
        assert "embedding_norm" in report.embedding_stats

    def test_no_embeddings_still_works(self):
        feat = _make_feat(with_embeddings=False)
        with patch("reasoning_agent.agent._call_ollama", return_value=_MOCK_RESPONSE):
            report = analyze(feat)
        assert isinstance(report, ReasoningReport)
        assert report.embedding_stats == {}

    def test_ollama_failure_returns_error_report(self):
        feat = _make_feat()
        with patch("reasoning_agent.agent._call_ollama", side_effect=RuntimeError("connection refused")):
            report = analyze(feat)
        assert report.error is not None
        assert "connection refused" in report.error

    def test_bad_json_returns_error_report(self):
        feat = _make_feat()
        with patch("reasoning_agent.agent._call_ollama", return_value="not json at all"):
            report = analyze(feat)
        assert report.error is not None

    def test_think_tags_stripped(self):
        feat = _make_feat()
        wrapped = f"<think>lots of reasoning here</think>\n{_MOCK_RESPONSE}"
        with patch("reasoning_agent.agent._call_ollama", return_value=wrapped):
            report = analyze(feat)
        assert report.error is None
        assert report.most_likely == "Normal EEG pattern"

    def test_to_dict_serializable(self):
        import json as json_mod
        feat = _make_feat()
        with patch("reasoning_agent.agent._call_ollama", return_value=_MOCK_RESPONSE):
            report = analyze(feat)
        json_mod.dumps(report.to_dict())
