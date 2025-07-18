# config.py

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

# Parametri per il modello LSTM
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.2
LOOK_BACK = 10  # Numero di epoche passate da considerare

# Parametri per l'addestramento
BATCH_SIZE = 32
EPOCHS = 50 # NOTA: Per un esperimento LOSO completo, potresti voler ridurre questo valore (es. 15-20) per velocizzare.
VALIDATION_SPLIT = 0.2
TRAIN_PREDICTION_GAP_MINUTES = 30 # Gap usato durante l'addestramento del modello

# NUOVO: Gap di previsione da valutare durante il test (in minuti)
EVALUATION_GAPS_MINUTES = [30, 60, 90, 120]

# Elettrodi da utilizzare
EEG_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']