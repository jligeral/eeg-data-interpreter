"""
scripts/prepare_lead_dataset.py
================================
Convert labeled EEG files into the LEAD memmap dataset format so you can
fine-tune the P-Base pretrained model on your own data.

Expected input layout
---------------------
Two directories: one for AD/dementia recordings, one for healthy controls.

    data/
        AD/     ← EEG files for Alzheimer's / dementia  (label = 1)
        HC/     ← EEG files for healthy controls         (label = 0)

Supported file types: .edf, .bdf, .fif, .set, .gdf, .cnt, .vhdr, .csv, .npz, .mat

Output layout (ready for LEAD fine-tuning)
------------------------------------------
    dataset/L400/<dataset_name>/
        meta.json       ← {"N": ..., "T": 400, "C": 19}
        X.dat           ← float32 memmap, shape (N, 400, 19)
        y.dat           ← float32 memmap, shape (N, 3)  [label, subject_id, sfreq]

Usage
-----
    python scripts/prepare_lead_dataset.py \\
        --ad_dir   data/AD \\
        --hc_dir   data/HC \\
        --name     MyDataset \\
        --out_dir  dataset/L400

    # then fine-tune (from project root):
    cd lead && python run.py \\
        --task_name finetune --model LEADv2 \\
        --data MultiDatasets --data_folder_list MyDataset \\
        --root_path ../dataset/L400/ \\
        --pretrain_model_path ../checkpoints/LEADv2/P-Base/checkpoint.pth \\
        --checkpoints ../checkpoints/LEADv2/finetune/ \\
        --num_class 2 --classify_choice ad_vs_hc \\
        --enc_in 19 \\
        --channel_names Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,Fz,Cz,Pz \\
        --montage_name standard_1020 \\
        --seq_len 400 --patch_len 50 --stride 50 \\
        --d_model 128 --d_ff 256 --n_heads 8 --e_layers 12 \\
        --dropout 0.1 --batch_size 16 --learning_rate 1e-4 \\
        --train_epochs 20 --sampling_rate_list 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make sure project root is on sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from preprocessor.preprocess import preprocess
from feature_extractor.windower import resample_and_window, LEAD_CHANNEL_ORDER, LEAD_WINDOW

_SUPPORTED = {".edf", ".bdf", ".fif", ".set", ".gdf", ".cnt", ".vhdr",
              ".csv", ".tsv", ".txt", ".npz", ".npy", ".mat"}

_LEAD_SFREQ = 200


def _collect_files(directory: Path) -> list[Path]:
    return sorted(f for f in directory.iterdir() if f.suffix.lower() in _SUPPORTED)


def _process_file(filepath: Path, subject_id: int, label: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Run one EEG file through the preprocessor + windower.

    Returns
    -------
    (X_subj, y_subj) where
        X_subj : (n_windows, 400, n_channels) float32
        y_subj : (n_windows, 3) float32  — [label, subject_id, sfreq]
    Returns None if the file could not be processed or produces no windows.
    """
    try:
        record = preprocess(str(filepath))
    except Exception as e:
        print(f"  [SKIP] {filepath.name}: preprocess failed — {e}")
        return None

    windows, ch_names, notes = resample_and_window(record)

    if windows.size == 0:
        print(f"  [SKIP] {filepath.name}: no windows produced ({'; '.join(notes)})")
        return None

    n_windows = windows.shape[0]

    # Pad or trim to exactly 19 channels in LEAD order
    n_ch = len(ch_names)
    if n_ch < len(LEAD_CHANNEL_ORDER):
        pad = np.zeros((n_windows, LEAD_WINDOW, len(LEAD_CHANNEL_ORDER) - n_ch), dtype=np.float32)
        windows = np.concatenate([windows, pad], axis=2)

    y_subj = np.zeros((n_windows, 3), dtype=np.float32)
    y_subj[:, 0] = label
    y_subj[:, 1] = subject_id
    y_subj[:, 2] = _LEAD_SFREQ

    print(f"  [OK]   {filepath.name}: subject {subject_id}, label {label}, "
          f"{n_windows} windows ({len(ch_names)} ch)")
    return windows, y_subj


def prepare(ad_dir: Path, hc_dir: Path, name: str, out_dir: Path) -> None:
    out_path = out_dir / name
    out_path.mkdir(parents=True, exist_ok=True)

    ad_files = _collect_files(ad_dir)
    hc_files = _collect_files(hc_dir)

    if not ad_files:
        sys.exit(f"No supported EEG files found in {ad_dir}")
    if not hc_files:
        sys.exit(f"No supported EEG files found in {hc_dir}")

    print(f"\nFound {len(ad_files)} AD files and {len(hc_files)} HC files.")
    print(f"Output: {out_path}\n")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    subject_id = 0

    for label, files in [(1, ad_files), (0, hc_files)]:
        tag = "AD" if label == 1 else "HC"
        print(f"Processing {tag} files…")
        for filepath in files:
            result = _process_file(filepath, subject_id, label)
            if result is not None:
                all_X.append(result[0])
                all_y.append(result[1])
                subject_id += 1

    if not all_X:
        sys.exit("No files were successfully processed. Check the input directories.")

    X_arr = np.concatenate(all_X, axis=0).astype(np.float32)   # (N, T, C)
    y_arr = np.concatenate(all_y, axis=0).astype(np.float32)   # (N, 3)

    N, T, C = X_arr.shape
    print(f"\nTotal segments : {N}")
    print(f"Window shape   : ({T}, {C})")
    print(f"Label counts   : HC={int((y_arr[:, 0] == 0).sum())}  AD={int((y_arr[:, 0] == 1).sum())}")

    # Write memmaps
    X_mem = np.memmap(out_path / "X.dat", dtype=np.float32, mode="w+", shape=(N, T, C))
    X_mem[:] = X_arr
    X_mem.flush()

    y_mem = np.memmap(out_path / "y.dat", dtype=np.float32, mode="w+", shape=(N, 3))
    y_mem[:] = y_arr
    y_mem.flush()

    # Write meta.json
    meta = {"N": N, "T": T, "C": C}
    with open(out_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset written to {out_path}/")
    print("  meta.json")
    print(f"  X.dat  — {X_arr.nbytes / 1e6:.1f} MB")
    print(f"  y.dat  — {y_arr.nbytes / 1e6:.1f} MB")
    print(f"\nNext step — fine-tune LEAD (run from project root):")
    print(f"""
  cd lead && python run.py \\
    --task_name finetune --model LEADv2 \\
    --data MultiDatasets --data_folder_list {name} \\
    --root_path ../dataset/L400/ \\
    --pretrain_model_path ../checkpoints/LEADv2/P-Base/checkpoint.pth \\
    --checkpoints ../checkpoints/LEADv2/finetune/ \\
    --num_class 2 --classify_choice ad_vs_hc \\
    --enc_in 19 \\
    --channel_names Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,Fz,Cz,Pz \\
    --montage_name standard_1020 \\
    --seq_len 400 --patch_len 50 --stride 50 \\
    --d_model 128 --d_ff 256 --n_heads 8 --e_layers 12 \\
    --dropout 0.1 --batch_size 16 --learning_rate 1e-4 \\
    --train_epochs 20 --sampling_rate_list 200

  # The fine-tuned checkpoint will be at:
  #   checkpoints/LEADv2/finetune/.../checkpoint.pth
  # Copy it to:
  #   checkpoints/LEADv2/P-Base/checkpoint.pth
  # (overwrite the pretrain checkpoint, or update _CHECKPOINT in lead_encoder.py)
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a labeled EEG dataset for LEAD fine-tuning."
    )
    parser.add_argument("--ad_dir",  required=True, type=Path,
                        help="Directory of AD/dementia EEG files (label=1)")
    parser.add_argument("--hc_dir",  required=True, type=Path,
                        help="Directory of healthy control EEG files (label=0)")
    parser.add_argument("--name",    required=True,
                        help="Dataset name (e.g. MyDataset) — used as folder name")
    parser.add_argument("--out_dir", default=Path("dataset/L400"), type=Path,
                        help="Output root directory (default: dataset/L400)")
    args = parser.parse_args()

    for d in (args.ad_dir, args.hc_dir):
        if not d.is_dir():
            sys.exit(f"Directory not found: {d}")

    prepare(args.ad_dir, args.hc_dir, args.name, args.out_dir)


if __name__ == "__main__":
    main()
