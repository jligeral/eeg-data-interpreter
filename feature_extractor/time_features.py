"""
time_features.py — Sample Entropy and Hjorth complexity parameters.

Both metrics characterise signal complexity in the time domain.
AD is broadly associated with reduced complexity: lower SampEn and lower
Hjorth complexity relative to healthy controls.

SampEn (m=2, r=0.2σ): measures irregularity; lower → more predictable signal.
Hjorth complexity: ratio of signal derivatives' mobility; lower → simpler waveform.
Hjorth mobility: sqrt(var(dx/dt) / var(x)); related to mean frequency.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from preprocessor.eeg_record import EEGRecord

_SAMPEN_M = 2
_SAMPEN_R_FACTOR = 0.2
_MAX_SAMPLES = 500  # truncate per channel before SampEn for speed

_EXPECTED_EEG = {
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
}


def _sample_entropy(x: np.ndarray, m: int, r_factor: float) -> float | None:
    """Sample entropy of a 1D signal via Chebyshev template matching."""
    x = x[:_MAX_SAMPLES].astype(np.float64)
    std = float(np.std(x, ddof=1))
    if std == 0:
        return None
    r = r_factor * std

    def _count(m_: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(x, m_)
        dists = cdist(templates, templates, metric="chebyshev")
        np.fill_diagonal(dists, np.inf)
        return int((dists < r).sum()) // 2

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return None
    val = -np.log(A / B)
    return float(val) if np.isfinite(val) else None


def _hjorth(x: np.ndarray) -> tuple[float, float] | None:
    """Return (mobility, complexity) or None if signal is flat."""
    dx = np.diff(x.astype(np.float64))
    ddx = np.diff(dx)
    var_x = float(np.var(x))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))
    if var_x == 0 or var_dx == 0:
        return None
    mobility = float(np.sqrt(var_dx / var_x))
    complexity = float(np.sqrt(var_ddx / var_dx) / mobility) if mobility > 0 else 0.0
    return mobility, complexity


def compute_time_features(record: EEGRecord) -> dict:
    """
    Compute SampEn and Hjorth parameters from preprocessed EEG.

    Returns dict with:
      sample_entropy    : mean SampEn across EEG channels (None on failure)
      hjorth_mobility   : mean Hjorth mobility across EEG channels
      hjorth_complexity : mean Hjorth complexity across EEG channels
      notes             : processing notes
    """
    result: dict = {
        "sample_entropy": None,
        "hjorth_mobility": None,
        "hjorth_complexity": None,
        "notes": [],
    }

    if record.raw is None:
        result["notes"].append("Time features skipped: raw signal missing.")
        return result

    raw = record.raw
    eeg_picks = [ch for ch in raw.ch_names if ch.upper() in {c.upper() for c in _EXPECTED_EEG}]
    if not eeg_picks:
        result["notes"].append("Time features skipped: no standard EEG channels found.")
        return result

    data_uv = raw.get_data(picks=eeg_picks) * 1e6  # (n_ch, n_samples)
    n_ch = data_uv.shape[0]

    # Sample Entropy
    se_vals = []
    for sig in data_uv:
        se = _sample_entropy(sig, _SAMPEN_M, _SAMPEN_R_FACTOR)
        if se is not None:
            se_vals.append(se)

    if se_vals:
        result["sample_entropy"] = round(float(np.mean(se_vals)), 4)
        result["notes"].append(
            f"Sample entropy: {result['sample_entropy']:.3f} "
            f"(mean across {len(se_vals)}/{n_ch} channels, "
            f"m={_SAMPEN_M}, r={_SAMPEN_R_FACTOR}×σ, "
            f"first {_MAX_SAMPLES} samples/channel)."
        )
    else:
        result["notes"].append("Sample entropy: could not be computed (flat or constant signal).")

    # Hjorth parameters
    mob_vals, comp_vals = [], []
    for sig in data_uv:
        hj = _hjorth(sig)
        if hj is not None:
            mob_vals.append(hj[0])
            comp_vals.append(hj[1])

    if mob_vals:
        result["hjorth_mobility"] = round(float(np.mean(mob_vals)), 4)
        result["hjorth_complexity"] = round(float(np.mean(comp_vals)), 4)
        result["notes"].append(
            f"Hjorth parameters — mobility: {result['hjorth_mobility']:.4f}, "
            f"complexity: {result['hjorth_complexity']:.4f} "
            f"(mean across {len(mob_vals)} channels)."
        )
    else:
        result["notes"].append("Hjorth parameters: could not be computed (flat channels).")

    return result
