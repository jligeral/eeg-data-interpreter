import mne
import numpy as np
from pathlib import Path


class UnsupportedFormatError(Exception):
    pass


class CorruptFileError(Exception):
    pass


def load_csv(filepath: str, sfreq: float):
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)

    if data.shape[0] > data.shape[1]:
        data = data.T

    ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


EXTENSION_MAP = {
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".fif": mne.io.read_raw_fif,
    ".set": mne.io.read_raw_eeglab,
    ".cnt": mne.io.read_raw_cnt,
    ".csv": load_csv,
}


def sniff_format(filepath: str):
    try:
        with open(filepath, "rb") as f:
            header = f.read(256)
    except Exception:
        return None

    if b"BIOSEMI" in header:
        return mne.io.read_raw_bdf

    if b"EDF" in header or b"0       " in header:
        return mne.io.read_raw_edf

    return None


def load_eeg(filepath: str, csv_sfreq=None):
    ext = Path(filepath).suffix.lower()

    loader = EXTENSION_MAP.get(ext) or sniff_format(filepath)

    if loader is None:
        raise UnsupportedFormatError(
            f"Unknown EEG format for file: {filepath}"
        )

    try:
        if ext == ".csv":
            if csv_sfreq is None:
                raise ValueError("CSV requires sampling frequency (csv_sfreq)")
            raw = loader(filepath, sfreq=csv_sfreq)
        else:
            raw = loader(filepath, preload=True)

        metadata = {
            "file_path": filepath,
            "file_format": ext,
            "sampling_rate": float(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
        }

        return raw, metadata

    except Exception as e:
        raise CorruptFileError(
            f"Failed to load EEG file: {filepath}\nReason: {e}"
        )
