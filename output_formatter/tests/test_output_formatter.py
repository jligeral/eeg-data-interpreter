"""
Tests for the output formatter module.
"""
import json
import pytest
import numpy as np

from feature_extractor.feature_record import FeatureRecord
from feature_extractor.windower import LEAD_CHANNEL_ORDER
from reasoning_agent.report import ReasoningReport, Hypothesis
from output_formatter.formatter import format_report
from output_formatter.eeg_report import EEGReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feat(
    quality="good",
    alpha_peak=10.0,
    theta_alpha=0.8,
    coherence=0.6,
    posterior_alpha=0.35,
    with_embeddings=True,
    confounds=None,
):
    feat = FeatureRecord(
        filepath="test.fif",
        file_format=".fif",
        data_quality=quality,
        channel_labels=list(LEAD_CHANNEL_ORDER),
        sampling_rate=256.0,
        confounds=confounds or [],
        alpha_peak_frequency=alpha_peak,
        theta_alpha_ratio=theta_alpha,
        posterior_alpha_power=posterior_alpha,
        posterior_alpha_coherence=coherence,
        coherence_pairs={"O1-O2": 0.6, "P3-P4": 0.65},
        band_power_relative={
            "delta": {"frontal": 0.06},
            "theta": {"frontal": 0.10},
            "alpha": {"frontal": 0.14},
            "beta":  {"frontal": 0.31},
        },
        band_power_absolute={},
        preprocessor_notes=["Bandpass filtered 0.5–45 Hz.", "Average reference applied."],
    )
    if with_embeddings:
        rng = np.random.default_rng(0)
        feat.lead_embeddings = rng.standard_normal((10, 19 * 128)).astype(np.float32)
        feat.lead_embeddings_available = True
        feat.n_windows = 10
    return feat


_DEFAULT_HYPOTHESIS = Hypothesis(
    name="Normal",
    description="All markers within normal range.",
    confidence=0.75,
    evidence_for=["Alpha peak at 10 Hz"],
    evidence_against=[],
)

_SENTINEL = object()


def _make_reasoning(
    hypotheses=_SENTINEL,
    most_likely="Normal EEG pattern",
    flags=None,
    next_steps=_SENTINEL,
    what_changes=None,
    caveats=None,
    error=None,
):
    return ReasoningReport(
        hypotheses=[_DEFAULT_HYPOTHESIS] if hypotheses is _SENTINEL else hypotheses,
        most_likely=most_likely,
        clinical_flags=flags or [],
        recommended_next_steps=["No immediate action required."] if next_steps is _SENTINEL else next_steps,
        what_would_change_mind=what_changes or ["Elevated theta/alpha ratio on repeat EEG"],
        caveats=caveats or [],
        embedding_stats={"embedding_norm": 22.0, "inter_window_variance": 0.22},
        model="qwen3:8b",
        error=error,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

class TestFormatReport:

    def test_returns_eeg_report(self):
        report = format_report(_make_feat(), _make_reasoning())
        assert isinstance(report, EEGReport)

    def test_all_four_sections_present(self):
        report = format_report(_make_feat(), _make_reasoning())
        assert report.objective_findings is not None
        assert report.differential_diagnosis is not None
        assert report.uncertainty is not None
        assert report.next_steps is not None

    def test_timestamp_populated(self):
        report = format_report(_make_feat(), _make_reasoning())
        assert report.generated_at.endswith("Z")

    def test_model_name_passed_through(self):
        report = format_report(_make_feat(), _make_reasoning())
        assert report.pipeline_model == "qwen3:8b"

    def test_reasoning_error_passed_through(self):
        r = _make_reasoning(error="connection refused")
        report = format_report(_make_feat(), r)
        assert report.reasoning_error == "connection refused"


# ---------------------------------------------------------------------------
# Objective findings
# ---------------------------------------------------------------------------

class TestObjectiveFindings:

    def test_biomarkers_list_populated(self):
        report = format_report(_make_feat(), _make_reasoning())
        assert len(report.objective_findings.biomarkers) >= 4

    def test_slowed_alpha_marked_abnormal(self):
        feat = _make_feat(alpha_peak=7.0)
        report = format_report(feat, _make_reasoning())
        apf = next(b for b in report.objective_findings.biomarkers
                   if "alpha peak" in b.name.lower())
        assert apf.abnormal is True

    def test_normal_alpha_not_abnormal(self):
        feat = _make_feat(alpha_peak=10.0)
        report = format_report(feat, _make_reasoning())
        apf = next(b for b in report.objective_findings.biomarkers
                   if "alpha peak" in b.name.lower())
        assert apf.abnormal is False

    def test_elevated_tar_marked_abnormal(self):
        feat = _make_feat(theta_alpha=1.4)
        report = format_report(feat, _make_reasoning())
        tar = next(b for b in report.objective_findings.biomarkers
                   if "theta" in b.name.lower())
        assert tar.abnormal is True

    def test_reduced_coherence_marked_abnormal(self):
        feat = _make_feat(coherence=0.2)
        report = format_report(feat, _make_reasoning())
        coh = next(b for b in report.objective_findings.biomarkers
                   if "coherence" in b.name.lower())
        assert coh.abnormal is True

    def test_confounds_passed_through(self):
        feat = _make_feat(confounds=["drowsiness_moderate"])
        report = format_report(feat, _make_reasoning())
        assert "drowsiness_moderate" in report.objective_findings.confounds

    def test_lead_stats_attached(self):
        report = format_report(_make_feat(with_embeddings=True), _make_reasoning())
        assert report.objective_findings.lead_embedding_stats.get("embedding_norm") == 22.0

    def test_band_power_preserved(self):
        report = format_report(_make_feat(), _make_reasoning())
        assert "delta" in report.objective_findings.band_power_relative


# ---------------------------------------------------------------------------
# Differential diagnosis
# ---------------------------------------------------------------------------

class TestDifferentialDiagnosis:

    def test_hypotheses_ranked_by_confidence(self):
        r = _make_reasoning(hypotheses=[
            Hypothesis("Low", "low conf", 0.3, [], []),
            Hypothesis("High", "high conf", 0.8, [], []),
        ])
        report = format_report(_make_feat(), r)
        ranks = [h.rank for h in report.differential_diagnosis.hypotheses]
        confs = [h.confidence for h in report.differential_diagnosis.hypotheses]
        assert ranks == [1, 2]
        assert confs[0] > confs[1]

    def test_clinical_impression_passed_through(self):
        r = _make_reasoning(most_likely="Possible early AD pattern")
        report = format_report(_make_feat(), r)
        assert report.differential_diagnosis.clinical_impression == "Possible early AD pattern"

    def test_confidence_labels_assigned(self):
        r = _make_reasoning(hypotheses=[
            Hypothesis("High", "", 0.8, [], []),
            Hypothesis("Mod",  "", 0.5, [], []),
            Hypothesis("Low",  "", 0.2, [], []),
        ])
        report = format_report(_make_feat(), r)
        labels = {h.name: h.confidence_label for h in report.differential_diagnosis.hypotheses}
        assert labels["High"] == "high"
        assert labels["Mod"]  == "moderate"
        assert labels["Low"]  == "low"

    def test_evidence_preserved(self):
        r = _make_reasoning(hypotheses=[
            Hypothesis("X", "desc", 0.7, ["evidence A"], ["counter B"]),
        ])
        report = format_report(_make_feat(), r)
        h = report.differential_diagnosis.hypotheses[0]
        assert "evidence A" in h.supporting_evidence
        assert "counter B" in h.contradicting_evidence


# ---------------------------------------------------------------------------
# Uncertainty profile
# ---------------------------------------------------------------------------

class TestUncertaintyProfile:

    def test_overall_confidence_from_top_hypothesis(self):
        r = _make_reasoning(hypotheses=[
            Hypothesis("A", "", 0.65, [], []),
            Hypothesis("B", "", 0.40, [], []),
        ])
        report = format_report(_make_feat(), r)
        assert report.uncertainty.overall_confidence == pytest.approx(0.65)

    def test_poor_quality_adds_limiting_factor(self):
        feat = _make_feat(quality="poor")
        report = format_report(feat, _make_reasoning())
        factors = " ".join(report.uncertainty.limiting_factors).lower()
        assert "poor" in factors

    def test_no_embeddings_flagged(self):
        feat = _make_feat(with_embeddings=False)
        report = format_report(feat, _make_reasoning())
        factors = " ".join(report.uncertainty.limiting_factors).lower()
        assert "lead" in factors

    def test_what_would_change_passed_through(self):
        r = _make_reasoning(what_changes=["CSF biomarker confirmation"])
        report = format_report(_make_feat(), r)
        assert "CSF biomarker confirmation" in report.uncertainty.what_would_change_assessment

    def test_zero_confidence_when_no_hypotheses(self):
        r = _make_reasoning(hypotheses=[])
        report = format_report(_make_feat(), r)
        assert report.uncertainty.overall_confidence == 0.0


# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------

class TestNextSteps:

    def test_actions_passed_through(self):
        r = _make_reasoning(next_steps=["Cognitive screening (MMSE)", "Structural MRI"])
        report = format_report(_make_feat(), r)
        assert "Cognitive screening (MMSE)" in report.next_steps.recommended_actions
        assert "Structural MRI" in report.next_steps.recommended_actions

    def test_empty_next_steps_handled(self):
        r = _make_reasoning(next_steps=[])
        report = format_report(_make_feat(), r)
        assert report.next_steps.recommended_actions == []


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

class TestOutputFormats:

    def test_to_dict_is_json_serialisable(self):
        report = format_report(_make_feat(), _make_reasoning())
        json.dumps(report.to_dict())

    def test_to_json_returns_string(self):
        report = format_report(_make_feat(), _make_reasoning())
        out = report.to_json()
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "objective_findings" in parsed
        assert "differential_diagnosis" in parsed
        assert "uncertainty" in parsed
        assert "next_steps" in parsed

    def test_to_text_returns_string(self):
        report = format_report(_make_feat(), _make_reasoning())
        text = report.to_text()
        assert isinstance(text, str)
        assert len(text) > 100

    def test_to_text_contains_all_section_headers(self):
        report = format_report(_make_feat(), _make_reasoning())
        text = report.to_text()
        assert "OBJECTIVE FINDINGS" in text
        assert "DIFFERENTIAL DIAGNOSIS" in text
        assert "UNCERTAINTY" in text
        assert "NEXT STEPS" in text

    def test_to_text_flags_abnormal_biomarkers(self):
        feat = _make_feat(alpha_peak=6.5, theta_alpha=1.5)
        report = format_report(feat, _make_reasoning())
        text = report.to_text()
        assert "ABNORMAL" in text

    def test_to_text_shows_reasoning_error(self):
        r = _make_reasoning(error="timeout")
        report = format_report(_make_feat(), r)
        text = report.to_text()
        assert "timeout" in text
