# data_loader.py

import os
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from config import EEG_CHANNELS, EPOCH_DURATION, LABEL_MAP, FS

def fetch_physionet_subjects(subjects, recording=[1]):
    """
    Scarica i dati per i soggetti specificati usando MNE e restituisce i percorsi dei file.

    Args:
        subjects (list): Una lista di interi che rappresentano i numeri dei soggetti da scaricare (es. [0, 1, 2]).
        recording (list): Una lista di interi che rappresentano le registrazioni da scaricare (di solito [1]).

    Returns:
        list: Una lista di tuple, dove ogni tupla contiene il percorso del file PSG e del file di annotazione.
    """
    print(f"Download dei dati per i soggetti: {subjects}...")
    # Scarica i dati (se non già presenti) e ottiene i percorsi
    paths = fetch_data(subjects=subjects, recording=recording, on_missing='warn')
    print("Download completato.")
    
    # fetch_data restituisce una lista di liste [[psg1, hyp1], [psg2, hyp2]]. 
    # La convertiamo in una lista di tuple per coerenza.
    subject_files = [(path[0], path[1]) for path in paths]
    return subject_files

def load_sleep_data(edf_path, annotation_path):
    """
    Carica i dati grezzi EEG e le etichette degli stadi del sonno da un singolo soggetto.
    """
    try:
        # Usa mne.io.read_raw che è più generico e non si basa sull'estensione del file
        raw = mne.io.read_raw(edf_path, preload=True, stim_channel=None)
        # MNE riconosce automaticamente il formato .hyp per le annotazioni
        annot = mne.read_annotations(annotation_path)
        raw.set_annotations(annot, emit_warning=False)
        
        event_id = {key: val for key, val in LABEL_MAP.items() if key in annot.description}
        
        events, _ = mne.events_from_annotations(
            raw, event_id=event_id, chunk_duration=EPOCH_DURATION
        )
        
        raw.pick_channels(EEG_CHANNELS)
        
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=0.0,
            tmax=EPOCH_DURATION - 1.0 / FS,
            baseline=None,
            preload=True
        )
        
        data = epochs.get_data()
        labels = epochs.events[:, -1]
        
        print(f"Caricato {os.path.basename(edf_path)}: {len(data)} epoche trovate.")
        return data, labels

    except Exception as e:
        print(f"Errore durante il caricamento del file {edf_path}: {e}")
        return None, None

if __name__ == '__main__':
    # Esempio: scarica i dati per i primi 2 soggetti del dataset
    subjects_to_load = [0, 1]
    all_files = fetch_physionet_subjects(subjects=subjects_to_load)

    if not all_files:
        print("Nessun file scaricato o trovato.")
    else:
        # Carica i dati per il primo soggetto come esempio
        first_subject_psg, first_subject_annot = all_files[0]
        print(f"\nTentativo di caricare i dati per il soggetto: {os.path.basename(first_subject_psg)}")
        
        data, labels = load_sleep_data(first_subject_psg, first_subject_annot)

        if data is not None and labels is not None:
            print("\n--- Esempio di Dati Caricati ---")
            print(f"Forma dei dati (epoche, canali, campioni): {data.shape}")
            print(f"Forma delle etichette: {labels.shape}")
