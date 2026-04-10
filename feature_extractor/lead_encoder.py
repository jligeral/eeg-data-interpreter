"""
lead_encoder.py — run LEAD inference on windowed EEG data.

Expects the LEAD source to be cloned into lead/ at the project root:
    git clone https://github.com/DL4mHealth/LEAD lead/

Pretrained P-Base weights go at:
    checkpoints/LEADv2/P-Base/checkpoint.pth

Run scripts/download_lead_weights.py to fetch the weights from Google Drive.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

from feature_extractor.feature_record import FeatureRecord

# Paths
_REPO_ROOT = Path(__file__).parent.parent
_LEAD_DIR = _REPO_ROOT / "lead"
_CHECKPOINT = _REPO_ROOT / "checkpoints" / "LEADv2" / "P-Base" / "checkpoint.pth"

# P-Base model configuration
_PBASE_CONFIG = dict(
    task_name="pretrain_lead",
    patch_len=50,
    stride=50,
    seq_len=400,
    enc_in=19,
    d_model=128,
    d_ff=256,
    n_heads=8,
    e_layers=12,
    dropout=0.1,
    output_attention=False,
    activation="gelu",
    num_class=2,
    channel_names="Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,Fz,Cz,Pz",
    montage_name="standard_1020",
    augmentations="none",
)

# Lazy imports — populated on first successful load
_torch = None
_Model = None


def _try_load_deps() -> tuple[bool, str]:
    """Import torch and LEAD model. Returns (success, reason_if_failed)."""
    global _torch, _Model

    # Already fully loaded
    if _torch is not None and _Model is not None:
        return True, ""

    try:
        import torch
        _torch = torch
    except ImportError:
        return False, "torch not installed. Run: pip install torch"

    # Look for models/LEADv2.py inside the lead/ directory
    if not (_LEAD_DIR / "models" / "LEADv2.py").exists():
        return False, (
            f"LEAD source not found at {_LEAD_DIR}/models/LEADv2.py. "
            "Run: git clone https://github.com/DL4mHealth/LEAD lead/"
        )

    if str(_LEAD_DIR) not in sys.path:
        sys.path.insert(0, str(_LEAD_DIR))

    try:
        from models.LEADv2 import Model
        _Model = Model
    except ImportError as e:
        return False, f"Failed to import LEAD model: {e}"

    return True, ""


def _load_model(channel_names: list[str]) -> tuple[object | None, bool, str]:
    """
    Load the LEADv2 model with pretrained weights.

    Returns (model, classifier_available, note).
    classifier_available is True only if the checkpoint contains trained
    classifier weights (i.e., a fine-tuned checkpoint, not just P-Base pretraining).
    """
    if not _CHECKPOINT.exists():
        # Build a default model so callers always get a model object back
        config = argparse.Namespace(**_PBASE_CONFIG)
        config.enc_in = len(channel_names)
        config.channel_names = ",".join(channel_names)
        return _Model(config).float(), False, (
            f"Checkpoint not found at {_CHECKPOINT}. "
            "Run: python scripts/download_lead_weights.py"
        )

    ckpt = _torch.load(str(_CHECKPOINT), map_location="cpu", weights_only=False)

    # Strip SWA double-module prefix if present (SWA wraps in module.module.*)
    new_sd = OrderedDict()
    for k, v in ckpt.items():
        if k == "n_averaged":
            continue
        key = k
        while key.startswith("module."):
            key = key[len("module."):]
        new_sd[key] = v

    # Auto-detect task type from checkpoint contents
    # Fine-tuned checkpoints contain "classifier.*"; pretrain checkpoints have "projection_head.*"
    has_classifier_in_ckpt = any("classifier" in k for k in new_sd)
    config = argparse.Namespace(**_PBASE_CONFIG)
    config.enc_in = len(channel_names)
    config.channel_names = ",".join(channel_names)
    config.task_name = "finetune" if has_classifier_in_ckpt else "pretrain_lead"

    model = _Model(config).float()
    missing, _ = model.load_state_dict(new_sd, strict=False)

    # Classifier is available if the model was built with finetune task and weights loaded
    classifier_available = has_classifier_in_ckpt and not any("classifier" in k for k in missing)

    return model, classifier_available, ""


def encode_with_lead(
    windows: np.ndarray,
    channel_names: list[str],
    record: FeatureRecord,
) -> FeatureRecord:
    """
    Run LEAD encoder on pre-windowed, normalized EEG data.

    Parameters
    ----------
    windows       : (n_windows, 400, n_channels) float32  — (B, T, C) for LEAD
    channel_names : channel names in the same order as the C axis
    record        : FeatureRecord to populate

    Returns
    -------
    record with lead_embeddings and optionally lead_ad_probability populated.
    """
    if windows.size == 0:
        record.processing_notes.append("LEAD encoding skipped: no windows to process.")
        return record

    ok, reason = _try_load_deps()
    if not ok:
        record.processing_notes.append(f"LEAD encoding skipped: {reason}")
        return record

    model, classifier_available, ckpt_note = _load_model(channel_names)
    if ckpt_note:
        record.processing_notes.append(f"LEAD weights: {ckpt_note}")

    try:
        device = "mps" if _torch.backends.mps.is_available() else "cpu"
        model = model.to(device).eval()

        n_windows = windows.shape[0]
        x = _torch.tensor(windows, dtype=_torch.float32).to(device)   # (B, T, C)
        fs = _torch.full((n_windows,), 200, dtype=_torch.float32).to(device)
        padding_mask = _torch.ones(n_windows, windows.shape[1], dtype=_torch.float32).to(device)

        # Hook the layer that receives the (B, enc_in * d_model) embedding as input.
        # pretrain_lead models have projection_head; finetune models have classifier.
        if hasattr(model, "projection_head"):
            hook_layer = model.projection_head
        elif hasattr(model, "classifier"):
            hook_layer = model.classifier
        else:
            raise AttributeError("LEAD model has neither projection_head nor classifier")

        captured = []
        def _hook(module, inp, out):
            captured.append(inp[0].detach().cpu())

        hook = hook_layer.register_forward_hook(_hook)

        with _torch.no_grad():
            # pretrain_lead forward returns (reprs_h, reprs_z); finetune returns logits
            result = model(x, padding_mask, None, None, fs, None)

        hook.remove()

        if isinstance(result, tuple):
            logits = None
        else:
            logits = result             # (B, num_class) — fine-tuned model

        # Embeddings: (B, enc_in * d_model) pre-projection representation
        embeddings = captured[0].numpy()                             # (B, 2432)
        record.lead_embeddings = embeddings
        record.lead_embeddings_available = True
        record.n_windows = n_windows
        record.processing_notes.append(
            f"LEAD encoded {n_windows} windows → embeddings shape {embeddings.shape}."
        )

        # AD probability — only meaningful if classifier was fine-tuned
        if classifier_available and logits is not None:
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=-1).cpu().numpy()          # (B, 2)
            ad_probs = probs[:, 1].tolist()                          # class 1 = AD
            record.lead_ad_per_window = [round(p, 4) for p in ad_probs]
            record.lead_ad_probability = round(float(np.mean(ad_probs)), 4)
            record.processing_notes.append(
                f"AD probability (fine-tuned classifier): {record.lead_ad_probability:.3f} "
                f"(mean across {n_windows} windows)."
            )
        else:
            record.processing_notes.append(
                "AD probability not available — P-Base checkpoint has no trained classifier. "
                "Fine-tune or provide a supervised checkpoint for AD probability scores."
            )

    except Exception as e:
        record.processing_notes.append(f"LEAD inference failed: {e}")
        record.lead_embeddings_available = False

    return record
