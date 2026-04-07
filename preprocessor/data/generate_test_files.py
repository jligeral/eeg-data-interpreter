"""
Generate synthetic EEG test files for preprocessor integration testing.
Run from the repo root: python preprocessor/data/generate_test_files.py
"""

import numpy as np
import mne
from pathlib import Path

mne.set_log_level("WARNING")

OUT_DIR = Path(__file__).parent / "generated"
OUT_DIR.mkdir(exist_ok=True)

SFREQ = 256
DURATION = 30  # seconds
N_SAMPLES = int(SFREQ * DURATION)
T = np.linspace(0, DURATION, N_SAMPLES)

CH_NAMES_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]


def base_eeg(ch_names, sfreq=SFREQ, duration=DURATION, seed=42):
    """White-noise baseline EEG with realistic µV amplitude."""
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration)
    data = rng.standard_normal((len(ch_names), n)) * 10e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def add_alpha(raw, channels, freq=10.0, amplitude_uv=25.0):
    """Add a sinusoidal alpha oscillation to specified channels."""
    n = raw.n_times
    t = np.linspace(0, raw.times[-1], n)
    for ch in channels:
        if ch in raw.ch_names:
            idx = raw.ch_names.index(ch)
            raw._data[idx] += amplitude_uv * 1e-6 * np.sin(2 * np.pi * freq * t)


def save_fif(raw, name):
    path = OUT_DIR / name
    raw.save(str(path), overwrite=True)
    print(f"  saved {path.name}")
    return path


def save_edf(raw, name):
    path = OUT_DIR / name
    mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True)
    print(f"  saved {path.name}")
    return path


def save_set(raw, name):
    path = OUT_DIR / name
    mne.export.export_raw(str(path), raw, fmt="eeglab", overwrite=True)
    print(f"  saved {path.name}")
    return path


def save_csv(raw, name):
    path = OUT_DIR / name
    header = ",".join(raw.ch_names)
    np.savetxt(str(path), raw.get_data().T * 1e6, delimiter=",",
               header=header, comments="")
    print(f"  saved {path.name}")
    return path


# ---------------------------------------------------------------------------

def gen_clean_full_montage():
    """All 19 10-20 channels, stable 10 Hz alpha, good signal. (.fif)"""
    print("Generating: clean_full_montage.fif")
    raw = base_eeg(CH_NAMES_10_20)
    add_alpha(raw, ["O1", "O2", "P3", "P4"])
    save_fif(raw, "clean_full_montage.fif")


def gen_missing_posterior():
    """O1 and O2 absent — should flag as poor quality. (.edf)"""
    print("Generating: missing_posterior.edf")
    ch_names = [ch for ch in CH_NAMES_10_20 if ch not in ("O1", "O2")]
    raw = base_eeg(ch_names)
    add_alpha(raw, ["P3", "P4"])
    save_edf(raw, "missing_posterior.edf")


def gen_flat_channel():
    """Cz is completely flat (dead electrode). (.fif)"""
    print("Generating: flat_channel.fif")
    raw = base_eeg(CH_NAMES_10_20)
    add_alpha(raw, ["O1", "O2"])
    idx = raw.ch_names.index("Cz")
    raw._data[idx, :] = 0.0
    save_fif(raw, "flat_channel.fif")


def gen_muscle_artifact():
    """High-frequency EMG-like noise in T3/T4 (temporal). (.fif)"""
    print("Generating: muscle_artifact.fif")
    raw = base_eeg(CH_NAMES_10_20)
    add_alpha(raw, ["O1", "O2"])
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(N_SAMPLES) * 80e-6
    for ch in ["T3", "T4"]:
        idx = raw.ch_names.index(ch)
        raw._data[idx] += noise
    save_fif(raw, "muscle_artifact.fif")


def gen_drowsy():
    """Alpha peak drifts from 10 Hz → 6 Hz over the recording. (.fif)"""
    print("Generating: drowsy.fif")
    raw = base_eeg(CH_NAMES_10_20)
    # Linearly sweep frequency from 10 Hz down to 6 Hz
    inst_freq = np.linspace(10.0, 6.0, N_SAMPLES)
    phase = np.cumsum(2 * np.pi * inst_freq / SFREQ)
    drifting_alpha = 25e-6 * np.sin(phase)
    for ch in ["O1", "O2"]:
        idx = raw.ch_names.index(ch)
        raw._data[idx] += drifting_alpha
    save_fif(raw, "drowsy.fif")


def gen_low_sfreq():
    """64 Hz sampling rate — below the 80 Hz minimum threshold. (.edf)"""
    print("Generating: low_sfreq.edf")
    raw = base_eeg(CH_NAMES_10_20, sfreq=64)
    add_alpha(raw, ["O1", "O2"])
    save_edf(raw, "low_sfreq.edf")


def gen_channel_aliases():
    """Uses T7/T8/P7/P8 naming instead of T3/T4/T5/T6. (.set)"""
    print("Generating: channel_aliases.set")
    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "O2",
    ]
    raw = base_eeg(ch_names)
    add_alpha(raw, ["O1", "O2"])
    save_set(raw, "channel_aliases.set")


def gen_csv():
    """Full montage exported as CSV (requires csv_sfreq at load time)."""
    print("Generating: clean_full_montage.csv")
    raw = base_eeg(CH_NAMES_10_20)
    add_alpha(raw, ["O1", "O2"])
    save_csv(raw, "clean_full_montage.csv")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Writing test EEG files to: {OUT_DIR}\n")
    gen_clean_full_montage()
    gen_missing_posterior()
    gen_flat_channel()
    gen_muscle_artifact()
    gen_drowsy()
    gen_low_sfreq()
    gen_channel_aliases()
    gen_csv()
    print(f"\nDone. {len(list(OUT_DIR.iterdir()))} files in {OUT_DIR}")
