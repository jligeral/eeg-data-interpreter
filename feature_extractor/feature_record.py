from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FeatureRecord:
    # Source metadata (passed through from preprocessor — no raw MNE object)
    filepath: str = ""
    file_format: str = ""
    data_quality: str = "unknown"
    confounds: list = field(default_factory=list)
    channel_labels: list = field(default_factory=list)
    sampling_rate: float = 0.0
    preprocessor_notes: list = field(default_factory=list)

    # REVE embeddings
    # Shape: (n_windows, 512) — one 512-dim vector per 10s window
    embeddings: Optional[np.ndarray] = None
    embeddings_available: bool = False
    n_windows: int = 0

    # Interpretable band features — relative power per region
    # Structure: {"delta": {"frontal": 0.12, "temporal": 0.10, ...}, ...}
    band_power_relative: dict = field(default_factory=dict)
    band_power_absolute: dict = field(default_factory=dict)

    # Key Alzheimer's biomarkers
    alpha_peak_frequency: Optional[float] = None   # Hz — slowing < 8 Hz is a red flag
    theta_alpha_ratio: Optional[float] = None      # elevated ratio → AD-like pattern
    posterior_alpha_power: Optional[float] = None  # relative power at O1/O2/P3/P4

    # Feature extraction notes
    processing_notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return {
            "filepath": self.filepath,
            "file_format": self.file_format,
            "data_quality": self.data_quality,
            "confounds": self.confounds,
            "channel_labels": self.channel_labels,
            "sampling_rate": self.sampling_rate,
            "preprocessor_notes": self.preprocessor_notes,
            "embeddings_available": self.embeddings_available,
            "n_windows": self.n_windows,
            "embeddings_shape": list(self.embeddings.shape) if self.embeddings is not None else None,
            "band_power_relative": self.band_power_relative,
            "band_power_absolute": self.band_power_absolute,
            "alpha_peak_frequency": self.alpha_peak_frequency,
            "theta_alpha_ratio": self.theta_alpha_ratio,
            "posterior_alpha_power": self.posterior_alpha_power,
            "processing_notes": self.processing_notes,
        }

    def summarize(self) -> str:
        quality_symbol = {"good": "✓", "moderate": "⚠", "poor": "✗"}.get(self.data_quality, "?")
        lines = [
            "Feature Record Summary",
            "══════════════════════════════════════",
            f"  File         : {self.filepath}",
            f"  Quality      : {quality_symbol} {self.data_quality.upper()}",
        ]

        if self.embeddings_available:
            lines.append(f"  REVE embeds  : {self.n_windows} windows × 512 dims")
        else:
            lines.append(f"  REVE embeds  : unavailable")

        if self.alpha_peak_frequency is not None:
            flag = " ← SLOWED" if self.alpha_peak_frequency < 8.0 else ""
            lines.append(f"  Alpha peak   : {self.alpha_peak_frequency:.2f} Hz{flag}")
        if self.theta_alpha_ratio is not None:
            flag = " ← ELEVATED" if self.theta_alpha_ratio > 1.0 else ""
            lines.append(f"  θ/α ratio    : {self.theta_alpha_ratio:.3f}{flag}")
        if self.posterior_alpha_power is not None:
            lines.append(f"  Post. alpha  : {self.posterior_alpha_power:.3f} (relative)")

        if self.band_power_relative:
            lines.append("  Band power (relative, by region):")
            for band, regions in self.band_power_relative.items():
                region_str = "  ".join(f"{r}={v:.3f}" for r, v in regions.items())
                lines.append(f"    {band:<8}: {region_str}")

        if self.confounds:
            lines.append(f"  Confounds ({len(self.confounds)}):")
            for c in self.confounds:
                lines.append(f"    • {c}")

        lines.append(f"  Processing notes ({len(self.processing_notes)}):")
        for note in self.processing_notes:
            lines.append(f"    • {note}")

        return "\n".join(lines)
