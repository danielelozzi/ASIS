# feature_extractor.py

import numpy as np
from scipy.signal import welch
from scipy.integrate import simps
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
    
    # Inizializza l'array che conterrà le feature
    psd_features = np.zeros((n_epochs, n_channels * n_bands))

    for epoch_idx in range(n_epochs):
        epoch_features = []
        for channel_idx in range(n_channels):
            # Estrae il segnale per l'epoca e il canale corrente
            signal = data_epochs[epoch_idx, channel_idx, :]
            
            # Calcola la PSD usando il metodo di Welch
            # nperseg è il numero di punti per segmento, lo impostiamo alla lunghezza dell'epoca
            freqs, psd = welch(signal, fs=FS, nperseg=n_samples)
            
            # Calcola la potenza per ogni banda di frequenza
            band_powers = []
            for band_name, (low_freq, high_freq) in BANDS.items():
                # Trova gli indici di frequenza che corrispondono alla banda
                freq_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
                
                # Calcola la potenza della banda usando l'integrazione (regola di Simpson)
                band_power = simps(psd[freq_indices], freqs[freq_indices])
                band_powers.append(band_power)
            
            epoch_features.extend(band_powers)
            
        psd_features[epoch_idx, :] = np.array(epoch_features)
        
    return psd_features

if __name__ == '__main__':
    # Esempio di utilizzo del feature extractor
    
    # NOTA: Sostituisci con il percorso reale del tuo dataset
    DATASET_PATH = './sleep-cassette' 

    print("--- Esempio di Estrazione Feature ---")

    if not os.path.isdir(DATASET_PATH):
        print(f"La directory del dataset non è stata trovata in: {DATASET_PATH}")
    else:
        all_files = data_loader.get_subject_files(DATASET_PATH)

        if not all_files:
            print(f"Nessun file PSG/Hypnogram trovato in {DATASET_PATH}")
        else:
            # Carica i dati per il primo soggetto
            psg_path, annot_path = all_files[0]
            data, labels = data_loader.load_sleep_data(psg_path, annot_path)

            if data is not None:
                # Estrai le feature PSD
                print("\nEstrazione delle feature PSD in corso...")
                features = extract_psd_features(data)
                
                print("\nEstrazione completata.")
                print(f"Forma dell'array di feature: {features.shape}")
                print(f"(Numero di epoche, Numero di feature)")
                
                # Le feature sono ordinate per canale, poi per banda.
                # Es: [ch1_delta, ch1_theta, ..., ch2_delta, ch2_theta, ...]
                print("\nEsempio di feature per la prima epoca:")
                print(features[0])

