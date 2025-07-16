# train.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

# Importa i nostri moduli personalizzati
import data_loader
import feature_extractor
import models
from config import (
    LOOK_BACK, 
    BATCH_SIZE, 
    EPOCHS, 
    VALIDATION_SPLIT, 
    LABEL_MAP,
    EEG_CHANNELS,
    BANDS
)

def prepare_sequences(features, labels, look_back, gap):
    """
    Trasforma i dati di feature e etichette in sequenze per l'LSTM.

    Args:
        features (np.ndarray): Array 2D di feature (n_epochs, n_features).
        labels (np.ndarray): Array 1D di etichette (n_epochs,).
        look_back (int): Numero di epoche passate da usare come input.
        gap (int): Il numero di epoche tra l'ultimo dato di input e il target.

    Returns:
        tuple: (X, y) dove X è l'array di sequenze e y l'array di etichette target.
    """
    X, y = [], []
    n_epochs = len(features)
    for i in range(n_epochs - look_back - gap):
        # La sequenza di input va da i a i+look_back
        sequence = features[i:i + look_back]
        # L'etichetta target è quella all'istante i+look_back+gap
        target_label = labels[i + look_back + gap]
        X.append(sequence)
        y.append(target_label)
    return np.array(X), np.array(y)

def train_general_model(dataset_path, models_output_dir='models'):
    """
    Addestra un modello generale usando i dati di tutti i soggetti.

    Args:
        dataset_path (str): Percorso della directory del dataset.
        models_output_dir (str): Directory dove salvare i modelli addestrati.
    """
    print("--- Inizio Addestramento Modello Generale ---")
    
    # 1. Carica e processa i dati di tutti i soggetti
    all_files = data_loader.get_subject_files(dataset_path)
    all_features = []
    all_labels = []

    for psg_path, annot_path in all_files:
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is not None:
            features = feature_extractor.extract_psd_features(data)
            all_features.append(features)
            all_labels.append(labels)
    
    # Concatena i dati di tutti i soggetti
    X_features = np.concatenate(all_features, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)
    
    # Normalizza le feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # --- Addestramento Modello Classico (Random Forest) ---
    print("\nAddestramento del classificatore classico (Random Forest)...")
    rf_model = models.get_classical_classifier()
    # Per il modello classico usiamo tutte le feature senza sequenziamento
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_scaled, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    rf_model.fit(X_train_rf, y_train_rf)
    accuracy = rf_model.score(X_test_rf, y_test_rf)
    print(f"Accuratezza del Random Forest sul set di test: {accuracy:.4f}")
    
    # --- Addestramento Modello LSTM ---
    print("\nPreparazione delle sequenze per LSTM...")
    # Esempio con un gap di 1 epoca (previsione dello stadio successivo)
    gap = 1 
    X_seq, y_seq = prepare_sequences(X_scaled, y_labels, LOOK_BACK, gap)
    
    if len(X_seq) == 0:
        print("Non ci sono abbastanza dati per creare sequenze con i parametri dati.")
        return

    print(f"Forma delle sequenze di input (X): {X_seq.shape}")
    print(f"Forma delle etichette target (y): {y_seq.shape}")

    # Crea il modello LSTM
    n_features = X_seq.shape[2]
    num_classes = len(np.unique(y_labels))
    lstm_model = models.create_lstm_model(input_shape=(LOOK_BACK, n_features), num_classes=num_classes)
    lstm_model.summary()
    
    print("\nAddestramento del modello LSTM...")
    history = lstm_model.fit(
        X_seq,
        y_seq,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

    # 4. Salva i modelli e lo scaler
    os.makedirs(models_output_dir, exist_ok=True)
    lstm_model.save(os.path.join(models_output_dir, 'general_lstm_model.h5'))
    joblib.dump(rf_model, os.path.join(models_output_dir, 'general_rf_model.pkl'))
    joblib.dump(scaler, os.path.join(models_output_dir, 'general_scaler.pkl'))
    
    print(f"\nModelli e scaler salvati nella directory '{models_output_dir}'.")


if __name__ == '__main__':
    # NOTA: Sostituisci con il percorso reale del tuo dataset
    DATASET_PATH = './sleep-cassette' 

    if not os.path.isdir(DATASET_PATH):
        print(f"La directory del dataset non è stata trovata in: {DATASET_PATH}")
        print("Assicurati che il percorso sia corretto.")
    else:
        # Avvia l'addestramento del modello generale
        train_general_model(DATASET_PATH)

