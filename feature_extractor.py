# feature_extractor.py

import numpy as np
from scipy.signal import welch
from scipy import integrate
from config import FS, BANDS

def extract_psd_features(data_epochs):
    n_epochs, n_channels, n_samples = data_epochs.shape
    n_bands = len(BANDS)
    psd_features = np.zeros((n_epochs, n_channels * n_bands))
    for epoch_idx in range(n_epochs):
        epoch_features = []
        for channel_idx in range(n_channels):
            signal = data_epochs[epoch_idx, channel_idx, :]
            freqs, psd = welch(signal, fs=FS, nperseg=n_samples)
            band_powers = []
            for band_name, (low_freq, high_freq) in BANDS.items():
                freq_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
                band_power = integrate.simpson(psd[freq_indices], freqs[freq_indices])
                band_powers.append(band_power)
            epoch_features.extend(band_powers)
        psd_features[epoch_idx, :] = np.array(epoch_features)
    return psd_features