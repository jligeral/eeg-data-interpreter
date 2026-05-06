from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
_CHECKPOINT = _REPO_ROOT / "checkpoints" / "labram" / "checkpoint-best.pth"

_torch = None
_create_model = None
_rearrange = None
_model = None

STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def get_input_chans(ch_names):
    input_chans = [0]  # CLS token
    for ch in ch_names:
        ch = str(ch).strip().upper()
        if ch not in STANDARD_1020:
            raise ValueError(f"Channel {ch} not found in LaBraM standard_1020 list.")
        input_chans.append(STANDARD_1020.index(ch) + 1)
    return input_chans

def _load_deps():
    global _torch, _create_model, _rearrange

    if _torch is not None:
        return

    import torch
    from timm.models import create_model
    from einops import rearrange
    import modeling_finetune  # registers LaBraM model

    _torch = torch
    _create_model = create_model
    _rearrange = rearrange


def _load_model():
    global _model

    if _model is not None:
        return _model

    _load_deps()

    if not _CHECKPOINT.exists():
        raise FileNotFoundError(f"LaBraM checkpoint not found: {_CHECKPOINT}")

    model = _create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=3,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=True,
    )

    checkpoint = _torch.load(
        str(_CHECKPOINT),
        map_location="cpu",
        weights_only=False
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    _model = model
    return _model


def encode_with_labram(windows, channel_names, record):
    """
    Expected input:
        windows shape = (n_windows, n_channels, 800)
        channel_names = ['FP1', 'FP2', ..., 'PZ']

    This matches our training setup:
        4-second windows at 200 Hz = 800 samples
    """

    try:
        _load_deps()
        model = _load_model()
        device = "cuda" if _torch.cuda.is_available() else "cpu"

        if windows is None or len(windows) == 0:
            record.processing_notes.append("LaBraM skipped: no windows found.")
            return record

        x_np = np.asarray(windows, dtype=np.float32)

        # If partner windower gives (B, T, C), convert to (B, C, T)
        if x_np.ndim == 3 and x_np.shape[1] == 800:
            x_np = np.transpose(x_np, (0, 2, 1))

        if x_np.ndim != 3:
            raise ValueError(f"Expected 3D windows, got shape {x_np.shape}")

        if x_np.shape[2] != 800:
            raise ValueError(
                f"LaBraM expects 800 timepoints per window from 4 sec at 200 Hz, got {x_np.shape[2]}"
            )

        x = _torch.tensor(x_np, dtype=_torch.float32).to(device) / 100.0

        # Match LaBraM training/eval code exactly:
        # (B, N, 800) -> (B, N, 4, 200)
        x = _rearrange(x, "B N (A T) -> B N A T", T=200)

        input_chans = get_input_chans(channel_names)

        with _torch.no_grad():
            logits = model(x, input_chans=input_chans)
            probs = _torch.softmax(logits, dim=1).cpu().numpy()

        mean_probs = probs.mean(axis=0)
        pred_idx = int(np.argmax(mean_probs))

        label_map = {
            0: "Healthy",
            1: "Alzheimer",
            2: "Other neurological",
        }

        record.labram_available = True
        record.labram_prob_healthy = float(mean_probs[0])
        record.labram_prob_alzheimers = float(mean_probs[1])
        record.labram_prob_other = float(mean_probs[2])
        record.labram_prediction = label_map[pred_idx]
        record.labram_per_window = probs.round(4).tolist()
        record.n_windows = int(x_np.shape[0])

        record.processing_notes.append(
            f"LaBraM processed {x_np.shape[0]} windows using 4-second, 200 Hz segments. "
            f"Prediction: {record.labram_prediction} "
            f"(Healthy={mean_probs[0]:.3f}, AD={mean_probs[1]:.3f}, Other={mean_probs[2]:.3f})."
        )

    except Exception as e:
        record.labram_available = False
        print(">>> LaBraM inference failed:", repr(e))
        record.processing_notes.append(f"LaBraM inference failed: {repr(e)}")

    return record