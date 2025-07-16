# data_loader.py

import os
import numpy as np
import mne
from config import EEG_CHANNELS, EPOCH_DURATION, LABEL_MAP, FS

def get_subject_files(dataset_path):
    """
    Trova tutte le coppie di file EDF e di annotazione in un dato percorso.

    Args:
        dataset_path (str): Il percorso della directory del dataset.

    Returns:
        list: Una lista di tuple, dove ogni tupla contiene il percorso
              del file EDF e del file di annotazione corrispondente.
    """
    subject_files = []
    # Cerca tutti i file di annotazione (Hypnogram)
    annotation_files = [f for f in os.listdir(dataset_path) if f.endswith('-Hypnogram.edf')]

    for annot_file in annotation_files:
        # Ricava il file PSG corrispondente
        base_name = annot_file.replace('-Hypnogram.edf', '')
        psg_file = base_name + '-PSG.edf'
        
        psg_path = os.path.join(dataset_path, psg_file)
        annot_path = os.path.join(dataset_path, annot_file)

        if os.path.exists(psg_path):
            subject_files.append((psg_path, annot_path))
            
    return subject_files

def load_sleep_data(edf_path, annotation_path):
    """
    Carica i dati grezzi EEG e le etichette degli stadi del sonno da un singolo soggetto.

    Args:
        edf_path (str): Percorso del file EDF contenente i dati PSG.
        annotation_path (str): Percorso del file di annotazione (ipnogramma).

    Returns:
        tuple: Una tupla contenente:
               - np.ndarray: Dati delle epoche (n_epochs, n_channels, n_samples).
               - np.ndarray: Etichette per ogni epoca.
    """
    try:
        # Carica il file EDF grezzo, escludendo i canali di stimolo
        raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None)
        
        # Carica le annotazioni dal file dell'ipnogramma
        annot = mne.read_annotations(annotation_path)
        
        # Applica le annotazioni ai dati grezzi
        raw.set_annotations(annot, emit_warning=False)
        
        # Mappa le annotazioni testuali a ID numerici
        event_id = {key: val for key, val in LABEL_MAP.items() if key in annot.description}
        
        # Estrae gli eventi dalle annotazioni
        events, _ = mne.events_from_annotations(
            raw, event_id=event_id, chunk_duration=EPOCH_DURATION
        )
        
        # Seleziona solo i canali EEG di interesse specificati in config.py
        raw.pick_channels(EEG_CHANNELS)
        
        # Crea le epoche di 30 secondi
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=0.0,
            tmax=EPOCH_DURATION - 1.0 / FS, # Durata esatta di 30s
            baseline=None,
            preload=True
        )
        
        # Estrae i dati e le etichette
        data = epochs.get_data() # (n_epochs, n_channels, n_times)
        labels = epochs.events[:, -1]
        
        print(f"Caricato {edf_path}: {len(data)} epoche trovate.")
        return data, labels

    except Exception as e:
        print(f"Errore durante il caricamento del file {edf_path}: {e}")
        return None, None

if __name__ == '__main__':
    # Esempio di utilizzo
    # NOTA: Sostituisci con il percorso reale del tuo dataset
    DATASET_PATH = './sleep-cassette' 

    if not os.path.isdir(DATASET_PATH):
        print(f"La directory del dataset non è stata trovata in: {DATASET_PATH}")
        print("Per favore, scarica il dataset 'sleep-edfx' da PhysioNet e decomprimilo.")
    else:
        # Ottieni la lista di tutti i file dei soggetti
        all_files = get_subject_files(DATASET_PATH)

        if not all_files:
            print(f"Nessun file PSG/Hypnogram trovato in {DATASET_PATH}")
        else:
            # Carica i dati per il primo soggetto come esempio
            first_subject_psg, first_subject_annot = all_files[0]
            
            print(f"Tentativo di caricare i dati per il soggetto: {first_subject_psg}")
            
            data, labels = load_sleep_data(first_subject_psg, first_subject_annot)

            if data is not None and labels is not None:
                print("\n--- Esempio di Dati Caricati ---")
                print(f"Forma dei dati (epoche, canali, campioni): {data.shape}")
                print(f"Forma delle etichette: {labels.shape}")
                print(f"Prime 5 etichette: {labels[:5]}")
                print(f"Stadi del sonno unici trovati (mappati): {np.unique(labels)}")
