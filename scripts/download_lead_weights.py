"""
Download LEAD P-Base pretrained weights from Google Drive.

Usage:
    python scripts/download_lead_weights.py

Weights are saved to:
    checkpoints/LEADv2/P-Base/checkpoint.pth

The original checkpoint path in the LEAD repo is:
    checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth

Google Drive folder:
    https://drive.google.com/drive/folders/1_XUfU3vZB40rjivkNYf8L2slCahXPo43
"""

import sys
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "LEADv2" / "P-Base"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pth"

# Direct file ID from the Google Drive folder
GDRIVE_FILE_ID = "1_XUfU3vZB40rjivkNYf8L2slCahXPo43"


def download():
    try:
        import gdown
    except ImportError:
        print("gdown not installed. Run: pip install gdown")
        sys.exit(1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_PATH.exists():
        print(f"Checkpoint already exists at {CHECKPOINT_PATH}")
        return

    print(f"Downloading LEAD P-Base weights to {CHECKPOINT_PATH} ...")
    print("Note: if the direct download fails, download manually from:")
    print("  https://drive.google.com/drive/folders/1_XUfU3vZB40rjivkNYf8L2slCahXPo43")
    print(f"  and place checkpoint.pth at: {CHECKPOINT_PATH}\n")

    try:
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            str(CHECKPOINT_PATH),
            quiet=False,
        )
        print(f"\nSaved to {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from the Google Drive link above.")
        sys.exit(1)


if __name__ == "__main__":
    download()
