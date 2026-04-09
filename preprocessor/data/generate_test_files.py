"""
Generate synthetic EEG test files in all supported formats.
Run from repo root: python preprocessor/data/generate_test_files.py
"""

import numpy as np
import mne
import scipy.io
from pathlib import Path

mne.set_log_level("WARNING")

OUT_DIR = Path(__file__).parent / "generated"
OUT_DIR.mkdir(exist_ok=True)

SFREQ = 256
DURATION = 30
N_SAMPLES = int(SFREQ * DURATION)
T = np.linspace(0, DURATION, N_SAMPLES)

CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_eeg(ch_names=CH_NAMES, sfreq=SFREQ, duration=DURATION, seed=42):
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration)
    data = rng.standard_normal((len(ch_names), n)) * 10e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def add_alpha(raw, channels, freq=10.0, amplitude_uv=25.0):
    n = raw.n_times
    t = np.linspace(0, raw.times[-1], n)
    for ch in channels:
        if ch in raw.ch_names:
            idx = raw.ch_names.index(ch)
            raw._data[idx] += amplitude_uv * 1e-6 * np.sin(2 * np.pi * freq * t)


def _save(path, label):
    print(f"  saved {path.name}")
    return path


def save_fif(raw, name):
    path = OUT_DIR / name
    raw.save(str(path), overwrite=True, verbose=False)
    return _save(path, name)


def save_edf(raw, name):
    path = OUT_DIR / name
    mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True, verbose=False)
    return _save(path, name)


def save_set(raw, name):
    path = OUT_DIR / name
    mne.export.export_raw(str(path), raw, fmt="eeglab", overwrite=True, verbose=False)
    return _save(path, name)


def save_vhdr(raw, name):
    path = OUT_DIR / name
    mne.export.export_raw(str(path), raw, fmt="brainvision", overwrite=True, verbose=False)
    return _save(path, name)


def save_csv(raw, name):
    path = OUT_DIR / name
    header = ",".join(raw.ch_names)
    np.savetxt(str(path), raw.get_data().T * 1e6, delimiter=",",
               header=header, comments="")
    return _save(path, name)


def save_npz(raw, name):
    path = OUT_DIR / name
    np.savez(
        str(path),
        data=raw.get_data(),          # (C, T) in volts
        sfreq=np.array(raw.info["sfreq"]),
        ch_names=np.array(raw.ch_names),
    )
    return _save(path, name)


def save_mat(raw, name):
    path = OUT_DIR / name
    data_uv = raw.get_data() * 1e6    # save in µV for realism
    scipy.io.savemat(str(path), {
        "data": data_uv,
        "srate": float(raw.info["sfreq"]),
        "ch_names": np.array(raw.ch_names, dtype=object),
    })
    return _save(path, name)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def gen_clean_fif():
    print("Generating: clean_full_montage.fif")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2", "P3", "P4"])
    save_fif(raw, "clean_full_montage.fif")


def gen_clean_edf():
    print("Generating: clean_full_montage.edf")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2", "P3", "P4"])
    save_edf(raw, "clean_full_montage.edf")


def gen_clean_set():
    print("Generating: clean_full_montage.set")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2", "P3", "P4"])
    save_set(raw, "clean_full_montage.set")


def gen_clean_vhdr():
    print("Generating: clean_full_montage.vhdr")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2", "P3", "P4"])
    save_vhdr(raw, "clean_full_montage.vhdr")


def gen_clean_csv():
    print("Generating: clean_full_montage.csv")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2", "P3", "P4"])
    save_csv(raw, "clean_full_montage.csv")


def gen_clean_npz():
    print("Generating: clean_full_montage.npz")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2", "P3", "P4"])
    save_npz(raw, "clean_full_montage.npz")


def gen_clean_mat():
    print("Generating: clean_full_montage.mat")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2", "P3", "P4"])
    save_mat(raw, "clean_full_montage.mat")


def gen_missing_posterior():
    print("Generating: missing_posterior.edf")
    ch_names = [ch for ch in CH_NAMES if ch not in ("O1", "O2")]
    raw = base_eeg(ch_names=ch_names)
    add_alpha(raw, ["P3", "P4"])
    save_edf(raw, "missing_posterior.edf")


def gen_flat_channel():
    print("Generating: flat_channel.fif")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2"])
    raw._data[raw.ch_names.index("Cz"), :] = 0.0
    save_fif(raw, "flat_channel.fif")


def gen_muscle_artifact():
    print("Generating: muscle_artifact.fif")
    raw = base_eeg()
    add_alpha(raw, ["O1", "O2"])
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(N_SAMPLES) * 80e-6
    for ch in ["T3", "T4"]:
        raw._data[raw.ch_names.index(ch)] += noise
    save_fif(raw, "muscle_artifact.fif")


def gen_drowsy():
    print("Generating: drowsy.fif")
    raw = base_eeg()
    inst_freq = np.linspace(10.0, 6.0, N_SAMPLES)
    phase = np.cumsum(2 * np.pi * inst_freq / SFREQ)
    drifting = 25e-6 * np.sin(phase)
    for ch in ["O1", "O2"]:
        raw._data[raw.ch_names.index(ch)] += drifting
    save_fif(raw, "drowsy.fif")


def gen_low_sfreq():
    print("Generating: low_sfreq.edf")
    raw = base_eeg(sfreq=64)
    add_alpha(raw, ["O1", "O2"])
    save_edf(raw, "low_sfreq.edf")


def gen_channel_aliases():
    print("Generating: channel_aliases.set")
    aliased = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7",  "C3",  "Cz", "C4", "T8",   # T7/T8 instead of T3/T4
        "P7",  "P3",  "Pz", "P4", "P8",   # P7/P8 instead of T5/T6
        "O1",  "O2",
    ]
    raw = base_eeg(ch_names=aliased)
    add_alpha(raw, ["O1", "O2"])
    save_set(raw, "channel_aliases.set")


if __name__ == "__main__":
    print(f"Writing test EEG files to: {OUT_DIR}\n")
    gen_clean_fif()
    gen_clean_edf()
    gen_clean_set()
    gen_clean_vhdr()
    gen_clean_csv()
    gen_clean_npz()
    gen_clean_mat()
    gen_missing_posterior()
    gen_flat_channel()
    gen_muscle_artifact()
    gen_drowsy()
    gen_low_sfreq()
    gen_channel_aliases()
    n = len(list(OUT_DIR.iterdir()))
    print(f"\nDone. {n} files in {OUT_DIR}")
