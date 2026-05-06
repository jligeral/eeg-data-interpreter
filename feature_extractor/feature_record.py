from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FeatureRecord:
    # Provenance (passed through from preprocessor)
    filepath: str = ""
    file_format: str = ""
    data_quality: str = "unknown"
    confounds: list = field(default_factory=list)
    channel_labels: list = field(default_factory=list)
    sampling_rate: float = 0.0
    preprocessor_notes: list = field(default_factory=list)

    #LaBraM
    labram_available: bool = False
    labram_prob_healthy: Optional[float] = None
    labram_prob_alzheimers: Optional[float] = None 
    labram_prob_other: Optional[float] = None
    labram_prediction: Optional[str] = None
    labram_per_window: list = field(default_factory=list)

    # LEAD embeddings
    # Shape: (n_windows, 2432) — per-channel CLS token, 19 channels × 128 dims
    lead_embeddings: Optional[np.ndarray] = None
    lead_embeddings_available: bool = False
    n_windows: int = 0

    # AD probability (populated only when a fine-tuned classifier is available)
    lead_ad_probability: Optional[float] = None       # mean across windows
    lead_ad_per_window: list = field(default_factory=list)  # per-window probabilities

    # Interpretable band features
    # {"delta": {"frontal": 0.12, ...}, ...}
    band_power_relative: dict = field(default_factory=dict)
    band_power_absolute: dict = field(default_factory=dict)

    # Key AD biomarkers
    alpha_peak_frequency: Optional[float] = None       # Hz — slowing < 8 Hz is a red flag
    theta_alpha_ratio: Optional[float] = None          # elevated > 1.0 → AD-like pattern
    posterior_alpha_power: Optional[float] = None      # relative power at posterior channels
    posterior_alpha_coherence: Optional[float] = None  # mean alpha coherence between posterior pairs
    coherence_pairs: dict = field(default_factory=dict)

    # Feature extraction notes
    processing_notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "file_format": self.file_format,
            "data_quality": self.data_quality,
            "confounds": self.confounds,
            "channel_labels": self.channel_labels,
            "sampling_rate": self.sampling_rate,
            "preprocessor_notes": self.preprocessor_notes,
            "lead_embeddings_available": self.lead_embeddings_available,
            "lead_embeddings_shape": list(self.lead_embeddings.shape) if self.lead_embeddings is not None else None,
            "n_windows": self.n_windows,
            "lead_ad_probability": self.lead_ad_probability,
            "lead_ad_per_window": self.lead_ad_per_window,
            "band_power_relative": self.band_power_relative,
            "band_power_absolute": self.band_power_absolute,
            "alpha_peak_frequency": self.alpha_peak_frequency,
            "theta_alpha_ratio": self.theta_alpha_ratio,
            "posterior_alpha_power": self.posterior_alpha_power,
            "posterior_alpha_coherence": self.posterior_alpha_coherence,
            "coherence_pairs": self.coherence_pairs,
            "processing_notes": self.processing_notes,
            "labram_available": self.labram_available,
            "labram_prob_healthy": self.labram_prob_healthy,
            "labram_prob_alzheimers": self.labram_prob_alzheimers,
            "labram_prob_other": self.labram_prob_other,
            "labram_prediction": self.labram_prediction,
            "labram_per_window": self.labram_per_window,
        }

    def summarize(self) -> str:
        quality_symbol = {"good": "✓", "moderate": "⚠", "poor": "✗"}.get(self.data_quality, "?")
        lines = [
            "Feature Record Summary",
            "══════════════════════════════════════",
            f"  File         : {self.filepath}",
            f"  Quality      : {quality_symbol} {self.data_quality.upper()}",
        ]

        # LaBraM summary (replaces LEAD)
        if getattr(self, "labram_available", False):
            lines.append(f"  LaBraM       : {self.n_windows} windows")

            if self.labram_prediction:
                lines.append(f"  Prediction   : {self.labram_prediction}")

            if self.labram_prob_healthy is not None:
                lines.append(f"  Healthy prob : {self.labram_prob_healthy:.3f}")

            if self.labram_prob_alzheimers is not None:
                flag = " ← ELEVATED" if self.labram_prob_alzheimers > 0.6 else ""
                lines.append(f"  AD prob      : {self.labram_prob_alzheimers:.3f}{flag}")

            if self.labram_prob_other is not None:
                lines.append(f"  Other prob   : {self.labram_prob_other:.3f}")

        else:
            lines.append("  LaBraM       : unavailable")

        if self.alpha_peak_frequency is not None:
            flag = " ← SLOWED" if self.alpha_peak_frequency < 8.0 else ""
            lines.append(f"  Alpha peak   : {self.alpha_peak_frequency:.2f} Hz{flag}")
        if self.theta_alpha_ratio is not None:
            flag = " ← ELEVATED" if self.theta_alpha_ratio > 1.0 else ""
            lines.append(f"  θ/α ratio    : {self.theta_alpha_ratio:.3f}{flag}")
        if self.posterior_alpha_power is not None:
            lines.append(f"  Post. alpha  : {self.posterior_alpha_power:.3f} (relative)")
        if self.posterior_alpha_coherence is not None:
            flag = " ← REDUCED" if self.posterior_alpha_coherence < 0.5 else ""
            lines.append(f"  Post. coher. : {self.posterior_alpha_coherence:.3f}{flag}")

        if self.band_power_relative:
            lines.append("  Band power (relative, by region):")
            for band, regions in self.band_power_relative.items():
                region_str = "  ".join(f"{r}={v:.3f}" for r, v in regions.items())
                lines.append(f"    {band:<8}: {region_str}")

        if self.confounds:
            lines.append(f"  Confounds ({len(self.confounds)}):")
            for c in self.confounds:
                lines.append(f"    • {c}")

        if self.processing_notes:
            lines.append(f"  Processing notes ({len(self.processing_notes)}):")
            for note in self.processing_notes:
                lines.append(f"    • {note}")

        return "\n".join(lines)
