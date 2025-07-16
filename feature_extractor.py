# feature_extractor.py

import numpy as np
from scipy.signal import welch
from scipy import integrate  # MODIFICA: Importa il modulo integrate
from config import FS, BANDS
import data_loader # Per l'esempio di utilizzo

def extract_psd_features(data_epochs):
    """
    Estrae le feature della Power Spectral Density (PSD) da epoche di dati EEG.

    Args:
        data_epochs (np.ndarray): Un array di dati con forma 
                                  (n_epochs, n_channels, n_samples).

    Returns:
        np.ndarray: Un array di feature con forma 
                    (n_epochs, n_channels * n_bands).
    """
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
                
                # MODIFICA: Usa integrate.simpson invece di simps
                band_power = integrate.simpson(psd[freq_indices], freqs[freq_indices])
                band_powers.append(band_power)
            
            epoch_features.extend(band_powers)
            
        psd_features[epoch_idx, :] = np.array(epoch_features)
        
    return psd_features

if __name__ == '__main__':
    import os
    # Esempio di utilizzo del feature extractor
    
    DATASET_PATH = './sleep-cassette' 

    print("--- Esempio di Estrazione Feature ---")

    if not os.path.isdir(DATASET_PATH):
        print(f"La directory del dataset non è stata trovata in: {DATASET_PATH}")
    else:
        all_files = data_loader.get_subject_files(DATASET_PATH)

        if not all_files:
            print(f"Nessun file PSG/Hypnogram trovato in {DATASET_PATH}")
        else:
            psg_path, annot_path = all_files[0]
            data, labels = data_loader.load_sleep_data(psg_path, annot_path)

            if data is not None:
                print("\nEstrazione delle feature PSD in corso...")
                features = extract_psd_features(data)
                
                print("\nEstrazione completata.")
                print(f"Forma dell'array di feature: {features.shape}")
                print(f"(Numero di epoche, Numero di feature)")
                
                print("\nEsempio di feature per la prima epoca:")
                print(features[0])
