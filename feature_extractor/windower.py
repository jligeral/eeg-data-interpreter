"""
windower.py — prepare EEG data for LEAD inference.

LEAD expects:
  - Sampling rate : 200 Hz
  - Window length : 400 samples (2 seconds)
  - Tensor shape  : (B, T, C) = (n_windows, 400, n_channels)
  - Normalisation : z-score per channel per window, clipped to ±10
"""

import numpy as np
import mne
from preprocessor.eeg_record import EEGRecord

LEAD_SFREQ = 200          # Hz
LEAD_WINDOW = 400         # samples  (2 s at 200 Hz)
LEAD_CLIP = 10.0          # z-score clip threshold

# LEAD's expected channel order (must match configs.channel_names)
LEAD_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1",  "O2",  "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz",  "Cz",  "Pz",
]


def resample_and_window(record: EEGRecord) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Prepare EEG data for LEAD inference.

    Steps:
      1. Resample to 200 Hz
      2. Reorder channels to LEAD's expected order (available channels only)
      3. Slice into non-overlapping 2-second windows
      4. Z-score normalize per channel per window, clip to ±10

    Returns
    -------
    windows      : np.ndarray, shape (n_windows, 400, n_channels), float32
                   — (B, T, C) as expected by LEAD
    channel_names: list of channel name strings in the order they appear in windows
    notes        : processing notes
    """
    notes = []

    if record.raw is None:
        notes.append("Windower skipped: raw signal missing.")
        return np.empty((0,), dtype=np.float32), [], notes

    raw = record.raw.copy()
    current_sfreq = float(raw.info["sfreq"])

    if current_sfreq != LEAD_SFREQ:
        raw.resample(LEAD_SFREQ, verbose=False)
        notes.append(f"Resampled {current_sfreq:.0f} Hz → {LEAD_SFREQ} Hz for LEAD.")
    else:
        notes.append(f"Sampling rate already {LEAD_SFREQ} Hz — no resampling needed.")

    # Reorder to LEAD's expected channel order (skip channels not present)
    available = [ch for ch in LEAD_CHANNEL_ORDER if ch in raw.ch_names]
    if not available:
        notes.append("Windower skipped: no standard 10-20 channels found.")
        return np.empty((0,), dtype=np.float32), [], notes

    raw.pick(available)
    data = raw.get_data()  # (n_channels, n_samples)
    n_channels, n_samples = data.shape

    if n_samples < LEAD_WINDOW:
        notes.append(
            f"Recording too short for LEAD ({n_samples} samples < {LEAD_WINDOW} required). "
            f"Need at least {LEAD_WINDOW / LEAD_SFREQ:.0f}s of data."
        )
        return np.empty((0,), dtype=np.float32), available, notes

    n_windows = n_samples // LEAD_WINDOW
    trimmed = data[:, : n_windows * LEAD_WINDOW]            # (C, n_windows * 400)
    windows = trimmed.reshape(n_channels, n_windows, LEAD_WINDOW)
    windows = windows.transpose(1, 2, 0)                    # (n_windows, 400, C) = (B, T, C)

    # Z-score per channel per window, clip to ±LEAD_CLIP
    mean = windows.mean(axis=1, keepdims=True)              # (B, 1, C)
    std  = windows.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-10, 1.0, std)
    windows = np.clip((windows - mean) / std, -LEAD_CLIP, LEAD_CLIP)

    notes.append(
        f"Windowed into {n_windows} × {LEAD_WINDOW}-sample windows "
        f"({LEAD_WINDOW / LEAD_SFREQ:.0f}s each, {n_channels} channels). "
        f"Z-scored and clipped to ±{LEAD_CLIP}."
    )

    return windows.astype(np.float32), available, notes
