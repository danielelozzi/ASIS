# Parametri per l'estrazione delle feature
FS = 100  # Frequenza di campionamento
EPOCH_DURATION = 30  # Durata di un'epoca in secondi
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'sigma': (12, 16),
    'beta': (16, 30)
}

# Mappatura delle etichette degli stadi del sonno
LABEL_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,  # N3 e N4 raggruppati
    'Sleep stage R': 4,
}
# Mappa inversa per i nomi delle classi nei grafici
CLASS_NAMES = {v: k.replace('Sleep stage ', '') for k, v in LABEL_MAP.items()}
# Gestisce le etichette duplicate (3 e 4 -> N3)
CLASS_NAMES[3] = "N3/N4" 
UNIQUE_CLASS_NAMES = sorted(list(set(CLASS_NAMES.values())), key=lambda x: list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(x)])


# Parametri per il modello LSTM
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LOOK_BACK = 10  # Numero di epoche passate da considerare (5 minuti)

# Parametri per l'addestramento
BATCH_SIZE = 32
EPOCHS = 20 # Ridotto per esperimenti più veloci, aumentare se necessario
VALIDATION_SPLIT = 0.2
TRAIN_PREDICTION_GAP_MINUTES = 30 # Gap usato durante l'addestramento

# Parametri per la valutazione
EVALUATION_GAPS_MINUTES = [30, 60, 90, 120]

# Elettrodi da utilizzare
EEG_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

# Directory di output
OUTPUTS_DIR = 'outputs'