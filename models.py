# models.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.ensemble import RandomForestClassifier

from config import LSTM_UNITS, LSTM_DROPOUT, LSTM_RECURRENT_DROPOUT, LABEL_MAP

def create_lstm_model(input_shape, num_classes):
    """
    Crea e compila un modello LSTM per la previsione degli stadi del sonno.

    Args:
        input_shape (tuple): La forma dei dati di input 
                             (es. (look_back, n_features)).
        num_classes (int): Il numero di stadi del sonno da classificare.

    Returns:
        tensorflow.keras.Model: Il modello LSTM compilato.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Strato LSTM per catturare le dipendenze temporali
        LSTM(
            units=LSTM_UNITS,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_sequences=False # Restituisce solo l'output dell'ultimo step
        ),
        
        # Strato fully-connected per la classificazione finale
        Dense(units=num_classes, activation='softmax')
    ])
    
    # Compila il modello
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Adatto per etichette intere
        metrics=['accuracy']
    )
    
    return model

def get_classical_classifier(n_estimators=100, random_state=42):
    """
    Restituisce un classificatore classico pre-configurato.

    Args:
        n_estimators (int): Il numero di alberi nella foresta.
        random_state (int): Seme per la riproducibilità.

    Returns:
        sklearn.ensemble.RandomForestClassifier: Un classificatore Random Forest.
    """
    # Il Random Forest è una scelta robusta per questo tipo di dati tabellari (le feature PSD)
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1 # Usa tutti i core disponibili
    )
    return classifier

if __name__ == '__main__':
    # Esempio di utilizzo e visualizzazione dei modelli
    
    # --- Esempio LSTM ---
    print("--- Architettura Modello LSTM ---")
    # Parametri di esempio
    look_back = 10  # Numero di epoche passate da considerare
    n_features = 10 # 2 canali * 5 bande di frequenza
    num_sleep_stages = len(LABEL_MAP)
    
    # Crea il modello LSTM
    lstm_model = create_lstm_model(
        input_shape=(look_back, n_features),
        num_classes=num_sleep_stages
    )
    
    # Stampa un riassunto dell'architettura
    lstm_model.summary()
    
    # --- Esempio Classificatore Classico ---
    print("\n\n--- Classificatore Classico ---")
    classical_model = get_classical_classifier()
    print("Modello scelto:", classical_model)
    print("I parametri possono essere modificati, es: get_classical_classifier(n_estimators=200)")

