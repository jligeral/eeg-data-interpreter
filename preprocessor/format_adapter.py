"""
format_adapter.py — load EEG files in all common formats into MNE Raw.

Supported formats
-----------------
MNE-native (no extra deps):
  .edf / .edf+   European Data Format
  .bdf           BioSemi Data Format
  .fif           MNE native
  .fif.gz        compressed MNE native
  .set           EEGLAB
  .gdf           General Data Format
  .cnt           Neuroscan
  .vhdr          BrainVision (.vhdr + .vmrk + .eeg)

Array/tabular (require sfreq kwarg unless header is present):
  .csv           comma-separated, optional channel-name header row
  .tsv / .txt    tab-separated, optional channel-name header row
  .npz           NumPy archive  — keys: data (C×T), sfreq, ch_names
  .npy           NumPy array    — shape (C, T); requires sfreq kwarg
  .mat           MATLAB v5/v7.3 — EEGLAB-style or raw matrix
"""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("WARNING")


class UnsupportedFormatError(Exception):
    pass


class CorruptFileError(Exception):
    pass


# ---------------------------------------------------------------------------
# MNE-native readers
# ---------------------------------------------------------------------------

_MNE_READERS = {
    ".edf":    mne.io.read_raw_edf,
    ".bdf":    mne.io.read_raw_bdf,
    ".fif":    mne.io.read_raw_fif,
    ".set":    mne.io.read_raw_eeglab,
    ".gdf":    mne.io.read_raw_gdf,
    ".cnt":    mne.io.read_raw_cnt,
    ".vhdr":   mne.io.read_raw_brainvision,
}


# ---------------------------------------------------------------------------
# Tabular loaders (CSV / TSV / TXT)
# ---------------------------------------------------------------------------

def _load_tabular(filepath: str, sfreq: float | None, delimiter: str) -> mne.io.RawArray:
    if sfreq is None:
        raise ValueError(
            f"sfreq is required for tabular format ({Path(filepath).suffix}). "
            "Pass sfreq=<value> to load_eeg()."
        )

    with open(filepath) as f:
        first_line = f.readline().strip()

    # Detect header: header cells contain letters, data cells are numeric
    cells = first_line.split(delimiter)
    has_header = any(re.search(r"[A-Za-z]", c) for c in cells)

    if has_header:
        import pandas as pd
        df = pd.read_csv(filepath, sep=delimiter)
        ch_names = list(df.columns)
        data = df.values.T.astype(np.float64)
    else:
        data = np.loadtxt(filepath, delimiter=delimiter)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] > data.shape[1]:
            data = data.T
        ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]

    # Convert µV → V if values look like microvolts (typical EEG range > 1 µV)
    if np.abs(data).max() > 0.01:
        data = data * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


# ---------------------------------------------------------------------------
# NumPy loaders
# ---------------------------------------------------------------------------

def _load_npz(filepath: str, sfreq: float | None) -> mne.io.RawArray:
    archive = np.load(filepath, allow_pickle=True)

    data = archive["data"]
    if data.shape[0] > data.shape[1]:
        data = data.T  # ensure (C, T)

    file_sfreq = float(archive["sfreq"]) if "sfreq" in archive else sfreq
    if file_sfreq is None:
        raise ValueError("NPZ file missing 'sfreq' key and no sfreq kwarg provided.")

    if "ch_names" in archive:
        ch_names = [str(c) for c in archive["ch_names"]]
    else:
        ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]

    if np.abs(data).max() > 0.01:
        data = data.astype(np.float64) * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=file_sfreq, ch_types="eeg")
    return mne.io.RawArray(data.astype(np.float64), info, verbose=False)


def _load_npy(filepath: str, sfreq: float | None, ch_names: list[str] | None) -> mne.io.RawArray:
    data = np.load(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] > data.shape[1]:
        data = data.T

    if sfreq is None:
        raise ValueError("sfreq is required for .npy files.")

    n_ch = data.shape[0]
    names = ch_names if ch_names and len(ch_names) == n_ch else [f"EEG{i + 1}" for i in range(n_ch)]

    if np.abs(data).max() > 0.01:
        data = data.astype(np.float64) * 1e-6

    info = mne.create_info(ch_names=names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data.astype(np.float64), info, verbose=False)


# ---------------------------------------------------------------------------
# MATLAB loader
# ---------------------------------------------------------------------------

def _load_mat(filepath: str, sfreq: float | None) -> mne.io.RawArray:
    # Try scipy (v5 .mat) first
    try:
        import scipy.io
        mat = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)

        # EEGLAB structure
        if "EEG" in mat:
            eeg = mat["EEG"]
            data = np.array(eeg.data)
            file_sfreq = float(eeg.srate)
            try:
                ch_names = [str(loc.labels) for loc in eeg.chanlocs]
            except Exception:
                ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]
        else:
            # Raw matrix — look for common key names
            data_key = next((k for k in ("data", "EEG", "eeg", "signal") if k in mat), None)
            if data_key is None:
                raise ValueError("Cannot locate data array in .mat file.")
            data = np.array(mat[data_key])
            if data.shape[0] > data.shape[1]:
                data = data.T

            sfreq_key = next((k for k in ("srate", "sfreq", "fs", "Fs", "SR") if k in mat), None)
            file_sfreq = float(mat[sfreq_key]) if sfreq_key else sfreq
            if file_sfreq is None:
                raise ValueError(".mat file missing sampling rate and no sfreq kwarg provided.")
            # Use saved channel names if present
            ch_key = next((k for k in ("ch_names", "chanlabels", "labels") if k in mat), None)
            if ch_key is not None:
                ch_names = [str(c).strip() for c in np.asarray(mat[ch_key]).flat]
            else:
                ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]

        if np.abs(data).max() > 0.01:
            data = data.astype(np.float64) * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=file_sfreq, ch_types="eeg")
        return mne.io.RawArray(data.astype(np.float64), info, verbose=False)

    except NotImplementedError:
        pass  # v7.3 — fall through to h5py

    # v7.3 HDF5-based .mat
    try:
        import h5py
        with h5py.File(filepath, "r") as f:
            data = np.array(f["data"] if "data" in f else f[list(f.keys())[0]])
            if data.shape[0] > data.shape[1]:
                data = data.T

            file_sfreq = float(np.array(f["srate"])) if "srate" in f else sfreq
            if file_sfreq is None:
                raise ValueError("v7.3 .mat missing srate and no sfreq kwarg provided.")

            ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]

        if np.abs(data).max() > 0.01:
            data = data.astype(np.float64) * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=file_sfreq, ch_types="eeg")
        return mne.io.RawArray(data.astype(np.float64), info, verbose=False)
    except Exception as e:
        raise CorruptFileError(f"Failed to load .mat file: {e}")


# ---------------------------------------------------------------------------
# Content sniffing fallback
# ---------------------------------------------------------------------------

def _sniff_format(filepath: str) -> str | None:
    """Detect format by file magic bytes when extension is missing/wrong."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(256)
    except Exception:
        return None

    if b"BIOSEMI" in header:
        return ".bdf"
    if b"0       " in header[:8] or b"EDF" in header[:3]:
        return ".edf"
    if header[:4] in (b"\x1f\x8b", b"MNE "):
        return ".fif"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_eeg(
    filepath: str,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    # Legacy alias kept for callers using the old kwarg name
    csv_sfreq: float | None = None,
) -> tuple[mne.io.Raw, dict]:
    """
    Load an EEG file and return (raw, metadata).

    Parameters
    ----------
    filepath  : path to the EEG file
    sfreq     : sampling rate (required for .csv, .tsv, .txt, .npy;
                optional for .npz and .mat if embedded in the file)
    ch_names  : channel names list (optional, used for .npy)
    csv_sfreq : legacy alias for sfreq (kept for backward compatibility)
    """
    if csv_sfreq is not None and sfreq is None:
        sfreq = csv_sfreq

    path = Path(filepath)
    ext = "".join(path.suffixes).lower()   # handles .fif.gz correctly
    if ext not in _MNE_READERS and len(path.suffixes) > 1:
        ext = path.suffix.lower()          # fall back to final suffix only

    try:
        # MNE-native formats
        if ext in _MNE_READERS:
            raw = _MNE_READERS[ext](filepath, preload=True, verbose=False)

        # Tabular formats
        elif ext in (".csv",):
            raw = _load_tabular(filepath, sfreq, delimiter=",")
        elif ext in (".tsv", ".txt"):
            raw = _load_tabular(filepath, sfreq, delimiter="\t")

        # Array formats
        elif ext == ".npz":
            raw = _load_npz(filepath, sfreq)
        elif ext == ".npy":
            raw = _load_npy(filepath, sfreq, ch_names)

        # MATLAB
        elif ext == ".mat":
            raw = _load_mat(filepath, sfreq)

        else:
            # Try content sniffing before giving up
            sniffed = _sniff_format(filepath)
            if sniffed and sniffed in _MNE_READERS:
                raw = _MNE_READERS[sniffed](filepath, preload=True, verbose=False)
            else:
                raise UnsupportedFormatError(
                    f"Unsupported EEG format '{ext}' for file: {filepath}\n"
                    f"Supported: {', '.join(sorted(set(list(_MNE_READERS) + ['.csv', '.tsv', '.txt', '.npz', '.npy', '.mat'])))}."
                )

        if not raw.preload:
            raw.load_data(verbose=False)

        # Ensure all channels are marked as EEG where type is unknown
        unknown = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types())
                   if kind not in ("eeg", "eog", "emg", "ecg", "stim")]
        if unknown:
            raw.set_channel_types({ch: "eeg" for ch in unknown}, verbose=False)

        metadata = {
            "file_format": ext,
            "sampling_rate": float(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
        }
        return raw, metadata

    except (UnsupportedFormatError, ValueError):
        raise
    except Exception as e:
        raise CorruptFileError(f"Failed to load '{filepath}': {e}") from e
