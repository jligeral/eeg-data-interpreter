import numpy as np
from scipy.signal import welch

try:
    _trapz = np.trapezoid   # NumPy 2.0+
except AttributeError:
    _trapz = np.trapz       # NumPy < 2.0
from itertools import combinations
import mne
from mne_connectivity import spectral_connectivity_epochs
from preprocessor.eeg_record import EEGRecord

# Frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

# 10-20 channel groupings by region
REGIONS = {
    "frontal":  {"Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"},
    "temporal": {"T3", "T4", "T5", "T6"},
    "central":  {"C3", "C4", "Cz"},
    "parietal": {"P3", "P4", "Pz"},
    "occipital": {"O1", "O2"},
}

POSTERIOR_CHANNELS = {"O1", "O2", "P3", "P4", "T5", "T6"}
ALPHA_RANGE = (8.0, 13.0)
THETA_RANGE = (4.0, 8.0)
ALPHA_PEAK_SEARCH_RANGE = (6.0, 13.0)
COHERENCE_EPOCH_DURATION = 2.0  # seconds


def compute_band_features(record: EEGRecord) -> dict:
    """
    Compute interpretable EEG features from the preprocessed record.

    Returns a dict with:
      band_power_relative  : relative power per band per region
      band_power_absolute  : absolute PSD (µV²/Hz) per band per region
      alpha_peak_frequency : posterior alpha peak in Hz
      theta_alpha_ratio    : theta / alpha power ratio (posterior channels)
      posterior_alpha_power: relative alpha power at posterior channels
      notes                : list of processing notes
    """
    result = {
        "band_power_relative": {},
        "band_power_absolute": {},
        "alpha_peak_frequency": None,
        "theta_alpha_ratio": None,
        "posterior_alpha_power": None,
        "psd_freqs": [],
        "psd_by_region": {},
        "notes": [],
    }

    if record.raw is None:
        result["notes"].append("Band feature extraction skipped: raw signal missing.")
        return result

    raw = record.raw
    sfreq = float(raw.info["sfreq"])
    ch_names_upper = {ch.upper(): ch for ch in raw.ch_names}

    data_uv = raw.get_data(picks="eeg") * 1e6  # convert to µV
    ch_names = [raw.ch_names[i] for i in range(data_uv.shape[0])]
    n_channels, n_samples = data_uv.shape

    nperseg = min(n_samples, int(sfreq * 4))
    freqs, psd = welch(data_uv, fs=sfreq, nperseg=nperseg)  # (n_ch, n_freqs)

    total_power = psd.sum(axis=1)  # (n_ch,) — sum across all freqs

    # --- Store PSD for visualization (0.5–40 Hz) ---
    vis_mask = (freqs >= 0.5) & (freqs <= 40.0)
    vis_freqs = freqs[vis_mask]
    result["psd_freqs"] = vis_freqs.tolist()
    psd_by_region: dict = {}
    for region, region_chs in REGIONS.items():
        idxs = [
            i for i, ch in enumerate(ch_names)
            if ch.upper() in {r.upper() for r in region_chs}
        ]
        if not idxs:
            continue
        region_psd = psd[idxs][:, vis_mask].mean(axis=0)
        area = float(_trapz(region_psd, vis_freqs))
        if area > 0:
            region_psd = region_psd / area
        psd_by_region[region] = region_psd.tolist()
    result["psd_by_region"] = psd_by_region

    # --- Per-band, per-region power ---
    for band_name, (lo, hi) in BANDS.items():
        band_idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(band_idx) == 0:
            continue

        band_power = psd[:, band_idx].sum(axis=1)  # (n_ch,) — sum within band

        abs_regions = {}
        rel_regions = {}

        for region, region_chs in REGIONS.items():
            idxs = [
                i for i, ch in enumerate(ch_names)
                if ch.upper() in {r.upper() for r in region_chs}
            ]
            if not idxs:
                continue
            abs_val = float(band_power[idxs].mean())
            total_val = float(total_power[idxs].mean())
            rel_val = abs_val / total_val if total_val > 0 else 0.0
            abs_regions[region] = round(abs_val, 4)
            rel_regions[region] = round(rel_val, 4)

        result["band_power_absolute"][band_name] = abs_regions
        result["band_power_relative"][band_name] = rel_regions

    # --- Alpha peak frequency (posterior channels) ---
    # Search 6–13 Hz to catch slowed alpha; flag if below 8 Hz.
    post_idxs = [
        i for i, ch in enumerate(ch_names)
        if ch.upper() in {c.upper() for c in POSTERIOR_CHANNELS}
    ]
    if post_idxs:
        search_idx = np.where(
            (freqs >= ALPHA_PEAK_SEARCH_RANGE[0]) & (freqs < ALPHA_PEAK_SEARCH_RANGE[1])
        )[0]
        if len(search_idx) > 0:
            post_psd = psd[post_idxs][:, search_idx].mean(axis=0)
            peak_freq = float(freqs[search_idx[np.argmax(post_psd)]])
            result["alpha_peak_frequency"] = round(peak_freq, 2)

            if peak_freq < 8.0:
                result["notes"].append(
                    f"Posterior alpha peak at {peak_freq:.2f} Hz — below 8 Hz threshold, "
                    f"consistent with slowing seen in Alzheimer's."
                )
            else:
                result["notes"].append(
                    f"Posterior alpha peak at {peak_freq:.2f} Hz — within normal range."
                )
    else:
        result["notes"].append(
            "Alpha peak frequency skipped: no posterior channels available."
        )

    # --- Theta/alpha ratio (posterior) ---
    if post_idxs:
        theta_idx = np.where((freqs >= THETA_RANGE[0]) & (freqs < THETA_RANGE[1]))[0]
        alpha_idx = np.where((freqs >= ALPHA_RANGE[0]) & (freqs < ALPHA_RANGE[1]))[0]
        if len(theta_idx) > 0 and len(alpha_idx) > 0:
            post_psd = psd[post_idxs]
            theta_power = post_psd[:, theta_idx].mean()
            alpha_power = post_psd[:, alpha_idx].mean()
            ratio = float(theta_power / alpha_power) if alpha_power > 0 else None
            if ratio is not None:
                result["theta_alpha_ratio"] = round(ratio, 4)
                if ratio > 1.0:
                    result["notes"].append(
                        f"Theta/alpha ratio {ratio:.3f} > 1.0 — elevated, "
                        f"consistent with AD-pattern slowing."
                    )
                else:
                    result["notes"].append(
                        f"Theta/alpha ratio {ratio:.3f} — within normal range."
                    )

    # --- Posterior alpha relative power ---
    if post_idxs:
        alpha_idx = np.where((freqs >= ALPHA_RANGE[0]) & (freqs < ALPHA_RANGE[1]))[0]
        if len(alpha_idx) > 0:
            post_psd = psd[post_idxs]
            post_alpha = float(post_psd[:, alpha_idx].sum(axis=1).mean())
            post_total = float(post_psd.sum(axis=1).mean())
            rel = post_alpha / post_total if post_total > 0 else 0.0
            result["posterior_alpha_power"] = round(rel, 4)

    result["notes"].insert(0, f"Band features computed from {n_channels} channels at {sfreq:.1f} Hz.")
    return result


def compute_posterior_coherence(record: EEGRecord) -> dict:
    """
    Compute mean alpha-band coherence between posterior channel pairs
    (O1, O2, P3, P4, T5, T6) using MNE-Connectivity.

    Returns a dict with:
      posterior_alpha_coherence : mean coherence across all posterior pairs
      coherence_pairs           : per-pair coherence values
      notes                     : processing notes
    """
    result = {
        "posterior_alpha_coherence": None,
        "coherence_pairs": {},
        "notes": [],
    }

    if record.raw is None:
        result["notes"].append("Coherence skipped: raw signal missing.")
        return result

    raw = record.raw
    ch_upper = {ch.upper(): ch for ch in raw.ch_names}
    available = [
        ch_upper[c.upper()]
        for c in POSTERIOR_CHANNELS
        if c.upper() in ch_upper
    ]

    if len(available) < 2:
        result["notes"].append(
            f"Coherence skipped: need ≥2 posterior channels, found {len(available)}."
        )
        return result

    raw_post = raw.copy().pick(available)
    sfreq = raw_post.info["sfreq"]

    min_duration = COHERENCE_EPOCH_DURATION * 2
    if raw_post.times[-1] < min_duration:
        result["notes"].append(
            f"Coherence skipped: recording too short "
            f"({raw_post.times[-1]:.1f}s < {min_duration:.0f}s needed for 2 epochs)."
        )
        return result

    events = mne.make_fixed_length_events(raw_post, duration=COHERENCE_EPOCH_DURATION)
    if len(events) < 2:
        result["notes"].append("Coherence skipped: recording too short for epoch-based coherence.")
        return result

    epochs = mne.Epochs(
        raw_post, events,
        tmin=0.0, tmax=COHERENCE_EPOCH_DURATION - 1.0 / sfreq,
        baseline=None, preload=True, verbose=False,
    )

    conn = spectral_connectivity_epochs(
        epochs,
        method="coh",
        fmin=ALPHA_RANGE[0],
        fmax=ALPHA_RANGE[1],
        faverage=True,
        verbose=False,
    )

    coh_values = conn.get_data(output="dense")[:, :, 0]  # (n_ch, n_ch)
    ch_list = epochs.ch_names
    pair_values = {
        f"{ch_list[i]}-{ch_list[j]}": round(float(coh_values[i, j]), 4)
        for i, j in combinations(range(len(ch_list)), 2)
    }

    if pair_values:
        mean_coh = float(np.mean(list(pair_values.values())))
        result["posterior_alpha_coherence"] = round(mean_coh, 4)
        result["coherence_pairs"] = pair_values
        result["notes"].append(
            f"Posterior alpha coherence (mean): {mean_coh:.3f} across {len(pair_values)} pairs."
        )
    else:
        result["notes"].append("Coherence: no valid pairs computed.")

    return result
