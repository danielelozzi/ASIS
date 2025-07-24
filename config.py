# config.py

# Parametri per l'estrazione delle feature
FS = 100
EPOCH_DURATION = 30 # Secondi per epoca
BANDS = {
    'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
    'sigma': (12, 16), 'beta': (16, 30)
}

# Elettrodi
EEG_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz'] # <-- Questa riga DEVE esserci

# Directory di output
OUTPUTS_DIR = 'outputs_subject_specific' # Nuova directory per distinguere i risultati

# Mappatura etichette
LABEL_MAP = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
}
CLASS_NAMES = {v: k.replace('Sleep stage ', '') for k, v in LABEL_MAP.items()}
CLASS_NAMES[3] = "N3/N4"
UNIQUE_CLASS_NAMES = sorted(list(set(CLASS_NAMES.values())), key=lambda x: list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(x)])


# Parametri modello LSTM (Resi "leggerissimi")
LSTM_UNITS = 32 # Ridotto da 64
LSTM_DROPOUT = 0.1 # Ridotto da 0.2
LOOK_BACK = 10  # 5 minuti di storico (10 epoche * 30 secondi/epoca = 300 secondi = 5 minuti)

# Parametri addestramento
BATCH_SIZE = 16 # Ridotto per pochi dati
EPOCHS = 10 # Ridotto per addestramento veloce
VALIDATION_SPLIT = 0.2 # Manteniamo una split per la validazione interna

# Nuovi parametri temporali per l'esperimento per soggetto
TRAINING_WINDOW_HOURS_PER_SUBJECT = 3 # Le prime 3 ore di sonno per il training
PREDICTION_INCREMENT_MINUTES = 1 # Previsione con incrementi di 1 minuto

# Gap di previsione per LSTM (in minuti, dopo la fine della finestra di training)
# Questo sarà ora generato dinamicamente in run_advanced_loso.py
# PREDICTION_TARGET_GAP_MINUTES = 1 # Non usato direttamente qui, ma la logica sarà simile per i vari gap

# Directory di output
OUTPUTS_DIR = 'outputs_subject_specific' # Nuova directory per distinguere i risultati