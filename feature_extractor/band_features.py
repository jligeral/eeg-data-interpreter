import numpy as np
from scipy.signal import welch
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
# Wider search range for alpha peak — catches slowing into slow-alpha territory
ALPHA_PEAK_SEARCH_RANGE = (6.0, 13.0)


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
