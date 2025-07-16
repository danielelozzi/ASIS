# data_loader.py

import os
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from config import EEG_CHANNELS, EPOCH_DURATION, LABEL_MAP, FS

def _setup_mne_data_on_drive():
    """
    Funzione ausiliaria per montare Google Drive e impostare la directory dei dati di MNE.
    Viene eseguita solo se lo script rileva di essere in un ambiente Google Colab.
    """
    try:
        # Questo import funziona solo su Colab
        from google.colab import drive
        
        print("Ambiente Colab rilevato. Tentativo di montare Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        
        # Percorso della cartella dei dati su Google Drive
        gdrive_mne_path = '/content/drive/MyDrive/mne_data'
        print(f"Il percorso per i dati MNE sarà: {gdrive_mne_path}")
        
        if not os.path.exists(gdrive_mne_path):
            os.makedirs(gdrive_mne_path)
            print(f"Directory creata.")
        else:
            print(f"La directory esiste già.")
            
        # Imposta la configurazione di MNE per usare la cartella su Drive
        mne.set_config('MNE_DATA', gdrive_mne_path, set_env=True)
        print(f"MNE è stato configurato per usare la directory su Google Drive.")

    except ImportError:
        # Se l'import di google.colab fallisce, significa che non siamo su Colab
        print("Ambiente non Colab rilevato. Si utilizzerà la directory predefinita di MNE.")
    except Exception as e:
        print(f"Si è verificato un errore durante il setup di Google Drive: {e}")


def fetch_physionet_subjects(subjects, recording=[1]):
    """
    Scarica i dati per i soggetti specificati usando MNE.
    Prima controlla e imposta la directory dei dati su Google Drive se in ambiente Colab.
    """
    # Esegue il setup di Google Drive all'inizio
    _setup_mne_data_on_drive()

    print(f"\nDownload (o caricamento dalla cache) dei dati per i soggetti: {subjects}...")
    # MNE userà automaticamente il percorso impostato da _setup_mne_data_on_drive
    paths = fetch_data(subjects=subjects, recording=recording, on_missing='warn')
    print("Operazione completata.")
    
    subject_files = [(path[0], path[1]) for path in paths]
    return subject_files

def load_sleep_data(edf_path, annotation_path):
    """
    Carica i dati grezzi EEG e le etichette degli stadi del sonno da un singolo soggetto.
    """
    try:
        raw = mne.io.read_raw(edf_path, preload=True, stim_channel=None)
        annot = mne.read_annotations(annotation_path)
        raw.set_annotations(annot, emit_warning=False)
        
        event_id = {key: val for key, val in LABEL_MAP.items() if key in annot.description}
        
        events, _ = mne.events_from_annotations(# data_loader.py

import os
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from config import EEG_CHANNELS, EPOCH_DURATION, LABEL_MAP, FS

def _setup_mne_data_on_drive():
    """
    Funzione ausiliaria per montare Google Drive e impostare la directory dei dati di MNE.
    Viene eseguita solo se lo script rileva di essere in un ambiente Google Colab.
    """
    # Controlla se siamo su Colab
    if 'google.colab' not in str(get_ipython()):
        print("Ambiente non Colab rilevato. Si utilizzerà la directory predefinita di MNE.")
        return

    try:
        from google.colab import drive
        
        print("Ambiente Colab rilevato. Tentativo di montare Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        
        gdrive_mne_path = '/content/drive/MyDrive/mne_data'
        
        # Se il montaggio ha successo, procedi a creare la directory e a configurare MNE
        if os.path.exists('/content/drive/MyDrive'):
            print(f"Il percorso per i dati MNE sarà: {gdrive_mne_path}")
            os.makedirs(gdrive_mne_path, exist_ok=True)
            mne.set_config('MNE_DATA', gdrive_mne_path, set_env=True)
            print(f"MNE è stato configurato per usare la directory su Google Drive.")
        else:
            # Se il montaggio sembra essere fallito, solleva un'eccezione per essere gestita sotto
            raise Exception("Montaggio di Google Drive fallito o directory non trovata.")

    except Exception as e:
        # Se qualcosa va storto, stampa un avviso e continua usando la memoria locale di Colab
        print(f"\nAVVISO: Si è verificato un errore durante il setup di Google Drive: {e}")
        print("L'addestramento procederà utilizzando la memoria temporanea di Colab.")
        print("I dati verranno riscaricati alla prossima esecuzione.\n")
        # Resetta la configurazione al default nel caso fosse stata impostata parzialmente
        mne.set_config('MNE_DATA', None)


def fetch_physionet_subjects(subjects, recording=[1]):
    """
    Scarica i dati per i soggetti specificati usando MNE.
    Prima controlla e imposta la directory dei dati su Google Drive se in ambiente Colab.
    """
    _setup_mne_data_on_drive()

    print(f"\nDownload (o caricamento dalla cache) dei dati per i soggetti: {subjects}...")
    paths = fetch_data(subjects=subjects, recording=recording, on_missing='warn')
    print("Operazione completata.")
    
    subject_files = [(path[0], path[1]) for path in paths]
    return subject_files

def load_sleep_data(edf_path, annotation_path):
    """
    Carica i dati grezzi EEG e le etichette degli stadi del sonno da un singolo soggetto.
    """
    try:
        raw = mne.io.read_raw(edf_path, preload=True, stim_channel=None)
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
    subjects_to_load = [0, 1]
    all_files = fetch_physionet_subjects(subjects=subjects_to_load)

    if not all_files:
        print("Nessun file scaricato o trovato.")
    else:
        first_subject_psg, first_subject_annot = all_files[0]
        print(f"\nTentativo di caricare i dati per il soggetto: {os.path.basename(first_subject_psg)}")
        
        data, labels = load_sleep_data(first_subject_psg, first_subject_annot)

        if data is not None and labels is not None:
            print("\n--- Esempio di Dati Caricati ---")
            print(f"Forma dei dati (epoche, canali, campioni): {data.shape}")
            print(f"Forma delle etichette: {labels.shape}")

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
        first_subject_psg, first_subject_annot = all_files[0]
        print(f"\nTentativo di caricare i dati per il soggetto: {os.path.basename(first_subject_psg)}")
        
        data, labels = load_sleep_data(first_subject_psg, first_subject_annot)

        if data is not None and labels is not None:
            print("\n--- Esempio di Dati Caricati ---")
            print(f"Forma dei dati (epoche, canali, campioni): {data.shape}")
            print(f"Forma delle etichette: {labels.shape}")
