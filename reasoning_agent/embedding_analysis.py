"""
Derive interpretable statistics from raw LEAD embeddings.

Without a fine-tuned classifier we can't output a probability, but we can
characterise *how unusual* the embedding looks via:

  inter_window_variance   — mean per-dimension variance across windows.
                            High variance → unstable / disrupted dynamics.
  embedding_norm          — L2 norm of the mean embedding (mean across windows).
  pca_top1_variance_ratio — fraction of total variance explained by the first
                            principal component. Low value → diffuse, uncorrelated
                            activation pattern (seen more in pathological EEG).
  channel_block_norms     — L2 norm per 128-dim channel block (19 channels).
                            Highlights which channels drive the embedding most.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
_D_MODEL = 128


def compute_embedding_stats(
    embeddings: np.ndarray,  # (n_windows, enc_in * d_model)
    channel_names: Optional[list[str]] = None,
) -> dict:
    """
    Parameters
    ----------
    embeddings    : (n_windows, enc_in * d_model) float32
    channel_names : channel names matching the enc_in axis (optional; used for
                    per-channel block norm labels)

    Returns
    -------
    dict with interpretable scalars and a per-channel breakdown.
    """
    if embeddings is None or embeddings.size == 0:
        return {}

    n_windows, total_dim = embeddings.shape
    enc_in = total_dim // _D_MODEL

    channels = (channel_names if channel_names else _CHANNEL_ORDER)[:enc_in]

    # --- Inter-window variance ---
    # Mean variance across embedding dimensions; high = more dynamic variation
    inter_window_var = float(embeddings.var(axis=0).mean())

    # --- Norm of mean embedding ---
    mean_emb = embeddings.mean(axis=0)
    emb_norm = float(np.linalg.norm(mean_emb))

    # --- PCA top-1 variance ratio ---
    # Use SVD on mean-centred matrix; first singular value²/total gives ratio
    centred = embeddings - mean_emb
    if n_windows >= 2:
        _, s, _ = np.linalg.svd(centred, full_matrices=False)
        pca_top1 = float((s[0] ** 2) / (s ** 2).sum()) if s.sum() > 0 else None
    else:
        pca_top1 = None

    # --- Per-channel block norms ---
    # Split embedding into enc_in blocks of d_model dims and compute L2 norm
    blocks = mean_emb.reshape(enc_in, _D_MODEL)
    block_norms = {ch: round(float(np.linalg.norm(blocks[i])), 3) for i, ch in enumerate(channels)}

    # Flag top-3 and bottom-3 channels by norm
    sorted_norms = sorted(block_norms.items(), key=lambda x: -x[1])
    top_channels = [ch for ch, _ in sorted_norms[:3]]
    low_channels = [ch for ch, _ in sorted_norms[-3:]]

    return {
        "n_windows": n_windows,
        "inter_window_variance": round(inter_window_var, 4),
        "embedding_norm": round(emb_norm, 3),
        "pca_top1_variance_ratio": round(pca_top1, 4) if pca_top1 is not None else None,
        "channel_block_norms": block_norms,
        "highest_activation_channels": top_channels,
        "lowest_activation_channels": low_channels,
    }
