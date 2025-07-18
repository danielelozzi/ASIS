# data_loader.py

import os
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from config import EEG_CHANNELS, EPOCH_DURATION, LABEL_MAP, FS

def fetch_physionet_subjects(subjects, recording=[1]):
    print(f"\nDownload (o caricamento) dati per i soggetti: {subjects}...")
    paths = fetch_data(subjects=subjects, recording=recording, on_missing='warn')
    print("Operazione completata.")
    return [(path[0], path[1]) for path in paths]

def load_sleep_data(edf_path, annotation_path, max_epochs=None):
    """
    Carica dati e etichette, con opzione per limitare il numero di epoche.
    """
    try:
        raw = mne.io.read_raw(edf_path, preload=True, stim_channel=None)
        annot = mne.read_annotations(annotation_path)
        raw.set_annotations(annot, emit_warning=False)
        
        event_id = {key: val for key, val in LABEL_MAP.items() if key in annot.description}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=EPOCH_DURATION)
        
        raw.pick_channels(EEG_CHANNELS)
        
        epochs = mne.Epochs(
            raw, events=events, event_id=event_id, tmin=0.0,
            tmax=EPOCH_DURATION - 1.0 / FS, baseline=None, preload=True
        )
        
        if max_epochs is not None:
            epochs = epochs[:max_epochs]

        data = epochs.get_data()
        labels = epochs.events[:, -1]
        
        print(f"Caricato {os.path.basename(edf_path)}: {len(data)} epoche.")
        return data, labels
    except Exception as e:
        print(f"Errore durante il caricamento di {edf_path}: {e}")
        return None, None