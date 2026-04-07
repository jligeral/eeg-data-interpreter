from dataclasses import dataclass, field, asdict
from typing import Optional
import mne

@dataclass
class EEGRecord:
    raw: Optional[mne.io.Raw] = None
    filepath: str = ""
    file_format: str = ""

    sampling_rate: float = 0.0
    n_channels: int = 0
    channel_labels: list = field(default_factory=list)

    missing_channels: list = field(default_factory=list)
    unknown_channels: list = field(default_factory=list)
    dropped_channels: list = field(default_factory=list)

    flat_channels: list = field(default_factory=list)
    high_amplitude_channels: list = field(default_factory=list)
    montage_ok: bool = True

    epochs: Optional[mne.Epochs] = None
    n_epochs_before: int = 0
    n_epochs_rejected: int = 0

    data_quality: str = "unknown"
    confounds: list = field(default_factory=list)

    processing_notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (excludes raw MNE objects)."""
        d = asdict(self)
        d.pop("raw", None)
        d.pop("epochs", None)
        return d

    def summarize(self) -> str:
        """Return a human-readable summary of the record."""
        quality_symbol = {"good": "✓", "moderate": "⚠", "poor": "✗"}.get(self.data_quality, "?")
        lines = [
            f"EEG Record Summary",
            f"══════════════════════════════════════",
            f"  File         : {self.filepath}",
            f"  Format       : {self.file_format or 'unknown'}",
            f"  Quality      : {quality_symbol} {self.data_quality.upper()}",
            f"  Sampling rate: {self.sampling_rate:.1f} Hz",
            f"  Channels     : {self.n_channels}",
        ]

        if self.missing_channels:
            lines.append(f"  Missing chs  : {', '.join(self.missing_channels)}")
        if self.flat_channels:
            lines.append(f"  Flat chs     : {', '.join(self.flat_channels)}")
        if self.high_amplitude_channels:
            lines.append(f"  High-amp chs : {', '.join(self.high_amplitude_channels)}")

        if self.confounds:
            lines.append(f"  Confounds ({len(self.confounds)}):")
            for c in self.confounds:
                lines.append(f"    • {c}")
        else:
            lines.append(f"  Confounds    : none")

        lines.append(f"  Processing notes ({len(self.processing_notes)}):")
        for note in self.processing_notes:
            lines.append(f"    • {note}")

        return "\n".join(lines)
