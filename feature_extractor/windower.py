"""
windower.py — prepare EEG data for LaBraM inference.

LaBraM setup used in this project:
  - Sampling rate : 200 Hz
  - Window length : 800 samples (4 seconds)
  - Window overlap: 400 samples (2 seconds)
  - Tensor shape  : (B, C, T) = (n_windows, n_channels, 800)
"""

import numpy as np
from preprocessor.eeg_record import EEGRecord

LABRAM_SFREQ = 200
LABRAM_WINDOW = 800
LABRAM_OVERLAP = 400

LABRAM_CHANNEL_ORDER = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "FZ", "CZ", "PZ",
]


def _norm_ch(ch: str) -> str:
    ch = str(ch).strip().upper()
    ch = ch.replace("EEG ", "")
    ch = ch.replace("-REF", "")
    ch = ch.replace("-LE", "")
    return ch


def resample_and_window(record: EEGRecord) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Prepare EEG data for LaBraM inference.

    Returns
    -------
    windows:
        np.ndarray, shape (n_windows, n_channels, 800)
    channel_names:
        list of channel names in the same order as windows
    notes:
        processing notes
    """
    notes = []

    if record.raw is None:
        notes.append("LaBraM windower skipped: raw signal missing.")
        return np.empty((0,), dtype=np.float32), [], notes

    raw = record.raw.copy()
    current_sfreq = float(raw.info["sfreq"])

    # Normalize channel labels to match training format
    rename_map = {}
    for ch in raw.ch_names:
        normed = _norm_ch(ch)
        if ch != normed:
            rename_map[ch] = normed
    if rename_map:
        raw.rename_channels(rename_map)

    if current_sfreq != LABRAM_SFREQ:
        raw.resample(LABRAM_SFREQ, verbose=False)
        notes.append(f"Resampled {current_sfreq:.0f} Hz → {LABRAM_SFREQ} Hz for LaBraM.")
    else:
        notes.append(f"Sampling rate already {LABRAM_SFREQ} Hz — no resampling needed.")

    available = [ch for ch in LABRAM_CHANNEL_ORDER if ch in raw.ch_names]

    if not available:
        notes.append("LaBraM windower skipped: no expected 10-20 EEG channels found.")
        return np.empty((0,), dtype=np.float32), [], notes

    raw.pick(available)
    data = raw.get_data() * 1e6  # volts → microvolts
    n_channels, n_samples = data.shape

    if n_samples < LABRAM_WINDOW:
        notes.append(
            f"Recording too short for LaBraM ({n_samples} samples < {LABRAM_WINDOW} required)."
        )
        return np.empty((0,), dtype=np.float32), available, notes

    step = LABRAM_WINDOW - LABRAM_OVERLAP
    starts = range(0, n_samples - LABRAM_WINDOW + 1, step)

    windows = []
    for start in starts:
        windows.append(data[:, start:start + LABRAM_WINDOW])

    windows = np.stack(windows, axis=0).astype(np.float32)

    notes.append(
        f"Windowed into {windows.shape[0]} LaBraM windows "
        f"({LABRAM_WINDOW} samples = 4s, overlap = 2s, {n_channels} channels)."
    )

    return windows, available, notes