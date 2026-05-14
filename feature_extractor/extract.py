from preprocessor.eeg_record import EEGRecord
from feature_extractor.feature_record import FeatureRecord
from feature_extractor.band_features import compute_band_features, compute_posterior_coherence
from feature_extractor.time_features import compute_time_features
from feature_extractor.windower import resample_and_window
from feature_extractor.lead_encoder import encode_with_lead


def extract(record: EEGRecord) -> FeatureRecord:
    """
    Full feature extraction pipeline.

    Steps:
      1. Populate FeatureRecord from EEGRecord metadata
      2. Compute interpretable band features (power, alpha peak, θ/α ratio)
      3. Compute posterior alpha coherence
      4. Compute time-domain complexity features (SampEn, Hjorth)
      5. Resample + window signal for LEAD (2s windows at 200 Hz)
      6. Run LEAD encoder → embeddings + AD probability (if fine-tuned)

    Parameters
    ----------
    record : preprocessed EEGRecord

    Returns
    -------
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

    # Step 1: Band power features
    band_result = compute_band_features(record)
    feat.band_power_relative = band_result["band_power_relative"]
    feat.band_power_absolute = band_result["band_power_absolute"]
    feat.alpha_peak_frequency = band_result["alpha_peak_frequency"]
    feat.theta_alpha_ratio = band_result["theta_alpha_ratio"]
    feat.posterior_alpha_power = band_result["posterior_alpha_power"]
    feat.psd_freqs = band_result.get("psd_freqs", [])
    feat.psd_by_region = band_result.get("psd_by_region", {})
    feat.processing_notes.extend(band_result["notes"])

    feat.waveform_pre = dict(record.snapshot_pre)
    feat.waveform_post = dict(record.snapshot_post)

    # Step 2: Posterior alpha coherence
    coh_result = compute_posterior_coherence(record)
    feat.posterior_alpha_coherence = coh_result["posterior_alpha_coherence"]
    feat.coherence_pairs = coh_result["coherence_pairs"]
    feat.processing_notes.extend(coh_result["notes"])

    # Step 3: Time-domain complexity features
    time_result = compute_time_features(record)
    feat.sample_entropy = time_result["sample_entropy"]
    feat.hjorth_mobility = time_result["hjorth_mobility"]
    feat.hjorth_complexity = time_result["hjorth_complexity"]
    feat.processing_notes.extend(time_result["notes"])

    # Step 5: Window for LEAD
    windows, channel_names, window_notes = resample_and_window(record)
    feat.processing_notes.extend(window_notes)

    # Step 6: LEAD encoding
    feat = encode_with_lead(windows, channel_names, feat)

    return feat
