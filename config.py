# config.py

# Parametri per l'estrazione delle feature
FS = 100
EPOCH_DURATION = 30 # Secondi per epoca
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

# Elettrodi
EEG_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']


# Parametri modello LSTM (Leggerissimi per addestramento veloce e generalizzabile)
LSTM_UNITS = 32
LSTM_DROPOUT = 0.1
LOOK_BACK = 10  # 5 minuti di storico (10 epoche * 30 secondi/epoca = 300 secondi = 5 minuti)

# Parametri MLP per combinare output LSTM e feature temporale
MLP_HIDDEN_SIZE = 16 # Dimensione del layer nascosto del MLP
MLP_DROPOUT = 0.1

# Parametri addestramento generale
BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Nuovi parametri per la suddivisione dei soggetti
TRAIN_SUBJECT_RATIO = 0.8 # 80% dei soggetti per il training
RANDOM_SEED = 42 # Per riproducibilità della split

# Nuovi parametri temporali per l'esperimento generale
# La finestra di training sarà sempre le prime X ore del sonno dei soggetti di training (3 ore).
# Questi sono i momenti nel tempo (in minuti dall'inizio della registrazione) a cui vogliamo predire l'epoca.
# La sequenza LSTM terminerà sempre all'epoca corrispondente a 3 ore dall'inizio del sonno.
# Il "time_feature" passato all'MLP sarà la differenza tra PREDICTION_TARGET_TIMES_MINUTES e l'inizio del sonno.
PREDICTION_TARGET_TIMES_MINUTES = [
    180 + 1,   # 3 ore e 1 minuto (dal punto di training)
    180 + 2,   # 3 ore e 2 minuti
    180 + 5,   # 3 ore e 5 minuti
    180 + 10,  # 3 ore e 10 minuti
    180 + 15,  # 3 ore e 15 minuti
    180 + 30,  # 3 ore e 30 minuti
    180 + 45,  # 3 ore e 45 minuti
    180 + 60,  # 4 ore
    180 + 90,  # 4.5 ore
    180 + 120, # 5.0 ore
    180 + 150, # 5.5 ore
    180 + 180, # 6.0 ore
    180 + 210, # 6.5 ore
    180 + 240, # 7.0 ore
    180 + 270, # 7.5 ore
    180 + 300, # 8.0 ore
    180 + 330, # 8.5 ore
    180 + 360, # 9.0 ore
    180 + 390, # 9.5 ore
    180 + 420, # 10.0 ore
    180 + 450, # 10.5 ore
    180 + 480, # 11.0 ore
    180 + 510, # 11.5 ore
    180 + 540, # 12.0 ore
]

# Gap di predizione per l'addestramento LSTM generale (in minuti)
# Il modello LSTM sarà addestrato a prevedere l'epoca che si trova a questa distanza
# *dalla fine della sua sequenza di input LOOK_BACK*.
# Per l'addestramento, useremo un gap piccolo (es. 1 minuto). Il MLP poi adatterà questo per i gap più grandi.
LSTM_TRAIN_PREDICTION_GAP_MINUTES = 1

# Directory di output
OUTPUTS_DIR = 'outputs_general_prediction_with_time_feature' # Nuova directory