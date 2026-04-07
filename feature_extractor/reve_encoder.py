import numpy as np
from feature_extractor.feature_record import FeatureRecord

REVE_BASE = "brain-bzh/reve-base"
REVE_POSITIONS = "brain-bzh/reve-positions"
EMBED_DIM = 512

try:
    import torch
    from transformers import AutoModel
    _DEPS_AVAILABLE = True
except ImportError:
    torch = None
    AutoModel = None
    _DEPS_AVAILABLE = False


def encode_with_reve(
    windows: np.ndarray,
    channel_names: list[str],
    record: FeatureRecord,
) -> FeatureRecord:
    """
    Run REVE on pre-windowed, normalized EEG data.

    Args:
        windows       : (n_windows, n_channels, 2000) float32 array
        channel_names : list of EEG channel name strings
        record        : FeatureRecord to populate

    Returns:
        record with embeddings field populated (or skipped gracefully)
    """
    if windows.size == 0:
        record.processing_notes.append("REVE encoding skipped: no windows to process.")
        record.embeddings_available = False
        return record

    import feature_extractor.reve_encoder as _self
    if _self.AutoModel is None or _self.torch is None:
        record.processing_notes.append(
            "REVE encoding skipped: torch or transformers not installed. "
            "Run: pip install torch transformers"
        )
        record.embeddings_available = False
        return record

    try:
        pos_bank = _self.AutoModel.from_pretrained(REVE_POSITIONS, trust_remote_code=True)
        model = _self.AutoModel.from_pretrained(REVE_BASE, trust_remote_code=True)
    except Exception as e:
        record.processing_notes.append(
            f"REVE model load failed (model may not be downloaded): {e}"
        )
        record.embeddings_available = False
        return record

    try:
        device = "cuda" if _self.torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        pos_bank = pos_bank.to(device).eval()

        n_windows = windows.shape[0]
        x = _self.torch.tensor(windows, dtype=_self.torch.float32).to(device)  # (W, C, 2000)

        with _self.torch.no_grad():
            positions = pos_bank(channel_names)                     # (C, 3)
            positions = positions.unsqueeze(0).expand(n_windows, -1, -1)  # (W, C, 3)
            output = model(x, positions)                            # (W, C*H, 512)

        # REVE output shape: (W, C, H, 512) where C=channels, H=patches per channel
        # Pool over both C and H → (W, 512)
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        else:
            hidden = output

        while hidden.dim() > 2:
            hidden = hidden.mean(dim=1)  # iteratively collapse all middle dims

        embeddings = hidden.cpu().numpy()  # (W, 512)

        record.embeddings = embeddings
        record.n_windows = n_windows
        record.embeddings_available = True
        record.processing_notes.append(
            f"REVE encoded {n_windows} windows → embeddings shape {embeddings.shape}."
        )

    except Exception as e:
        record.processing_notes.append(f"REVE inference failed: {e}")
        record.embeddings_available = False

    return record
