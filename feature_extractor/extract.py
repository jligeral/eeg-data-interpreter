from preprocessor.eeg_record import EEGRecord
from feature_extractor.feature_record import FeatureRecord
from feature_extractor.resampler import resample_and_window
from feature_extractor.band_features import compute_band_features
from feature_extractor.reve_encoder import encode_with_reve


def extract(record: EEGRecord) -> FeatureRecord:
    """
    Full feature extraction pipeline.

    Steps:
      1. Populate FeatureRecord from EEGRecord metadata
      2. Compute interpretable band features
      3. Resample + window signal for REVE
      4. Run REVE encoder

    Args:
        record: preprocessed EEGRecord

    Returns:
        FeatureRecord ready for the Reasoning Agent
    """
    feat = FeatureRecord(
        filepath=record.filepath,
        file_format=record.file_format,
        data_quality=record.data_quality,
        confounds=list(record.confounds),
        channel_labels=list(record.channel_labels),
        sampling_rate=record.sampling_rate,
        preprocessor_notes=list(record.processing_notes),
    )

    # Step 1: Interpretable band features
    band_result = compute_band_features(record)
    feat.band_power_relative = band_result["band_power_relative"]
    feat.band_power_absolute = band_result["band_power_absolute"]
    feat.alpha_peak_frequency = band_result["alpha_peak_frequency"]
    feat.theta_alpha_ratio = band_result["theta_alpha_ratio"]
    feat.posterior_alpha_power = band_result["posterior_alpha_power"]
    feat.processing_notes.extend(band_result["notes"])

    # Step 2: Resample + window for REVE
    windows, channel_names, resample_notes = resample_and_window(record)
    feat.processing_notes.extend(resample_notes)

    # Step 3: REVE encoding
    feat = encode_with_reve(windows, channel_names, feat)

    return feat
