"""
scripts/download_demo_data.py
==============================
Download one AD and one HC subject from the ADFTD dataset (OpenNeuro ds004504)
for use as real demo/test files.

Output:
    data/demo/AD/sub-073_task-Rest_eeg.edf   ← Alzheimer's subject
    data/demo/HC/sub-001_task-Rest_eeg.edf   ← Healthy control

Usage:
    python scripts/download_demo_data.py
"""

from pathlib import Path
import shutil
import sys

try:
    import openneuro
except ImportError:
    sys.exit("openneuro-py not installed. Run: pip install openneuro-py")

_ROOT      = Path(__file__).parent.parent
_DATA_DIR  = _ROOT / "data" / "demo"
_AD_DIR    = _DATA_DIR / "AD"
_HC_DIR    = _DATA_DIR / "HC"
_TMP_DIR   = _ROOT / "data" / "_openneuro_tmp"

# ADFTD dataset on OpenNeuro (ds004504)
# participants.tsv groups:
#   A = Alzheimer's Disease  (sub-001 – sub-036)
#   C = Healthy Control      (sub-037 – sub-065)
#   F = Frontotemporal Dementia (sub-066 – sub-088)
# Files are EEGLAB .set format, task name "eyesclosed".
# sub-001: AD, MMSE=16. sub-037: HC, MMSE=30.
_DATASET     = "ds004504"
_AD_SUBJECT  = "sub-001"
_HC_SUBJECT  = "sub-037"
_TASK        = "eyesclosed"
_EXT         = ".set"


def _find_set(subject_dir: Path) -> Path | None:
    hits = list(subject_dir.rglob(f"*task-{_TASK}*_eeg{_EXT}"))
    return hits[0] if hits else None


def download_subject(subject: str, label_dir: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    tmp = _TMP_DIR / subject
    tmp.mkdir(parents=True, exist_ok=True)

    pattern = f"{subject}/eeg/*task-{_TASK}*_eeg{_EXT}"
    print(f"Downloading {subject} ({pattern}) …")
    openneuro.download(
        dataset=_DATASET,
        target_dir=tmp,
        include=[pattern],
        verify_hash=False,
    )

    eeg_file = _find_set(tmp)
    if eeg_file is None:
        print(f"  [WARN] No file found for {subject} — check the dataset structure.")
        return

    dest = label_dir / eeg_file.name
    shutil.copy(eeg_file, dest)
    print(f"  -> {dest}")


def main() -> None:
    print(f"Dataset : {_DATASET} (ADFTD resting-state EEG)")
    print(f"Output  : {_DATA_DIR}\n")

    download_subject(_AD_SUBJECT, _AD_DIR)
    download_subject(_HC_SUBJECT, _HC_DIR)

    # Clean up temp downloads
    if _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR)

    print("\nDone. Files saved to:")
    for f in sorted(_DATA_DIR.rglob("*.edf")):
        print(f"  {f.relative_to(_ROOT)}")
    print("\nRun `streamlit run app.py` and select these files from the sidebar.")


if __name__ == "__main__":
    main()
