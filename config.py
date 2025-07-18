# config.py

# Parametri per l'estrazione delle feature
FS = 100
EPOCH_DURATION = 30
BANDS = {
    'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
    'sigma': (12, 16), 'beta': (16, 30)
}

# Mappatura etichette
LABEL_MAP = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
}
CLASS_NAMES = {v: k.replace('Sleep stage ', '') for k, v in LABEL_MAP.items()}
CLASS_NAMES[3] = "N3/N4"
UNIQUE_CLASS_NAMES = sorted(list(set(CLASS_NAMES.values())), key=lambda x: list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(x)])

# Parametri modello LSTM
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LOOK_BACK = 10  # 5 minuti di storico

# Parametri addestramento
BATCH_SIZE = 32
EPOCHS = 15 # Ridotto per la complessità dell'esperimento
VALIDATION_SPLIT = 0.2

# NUOVO: Finestre temporali per l'addestramento incrementale (in minuti)
TRAINING_WINDOWS_MINUTES = [30, 60, 90, 120]

# NUOVO: Gap di predizione per LSTM (in minuti, dopo la fine della finestra di training)
PREDICTION_TARGET_GAP_MINUTES = 1 # Predici 1 minuto dopo la fine dei dati di training

# Elettrodi
EEG_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

# Directory di output
OUTPUTS_DIR = 'outputs_advanced'