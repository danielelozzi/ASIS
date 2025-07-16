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
    """
    X, y = [], []
    n_epochs = len(features)
    for i in range(n_epochs - look_back - gap):
        sequence = features[i:i + look_back]
        target_label = labels[i + look_back + gap]
        X.append(sequence)
        y.append(target_label)
    return np.array(X), np.array(y)

def train_general_model(subjects_to_load, models_output_dir='models'):
    """
    Addestra un modello generale usando i dati dei soggetti specificati.

    Args:
        subjects_to_load (list): Lista di interi dei soggetti da scaricare e usare.
        models_output_dir (str): Directory dove salvare i modelli addestrati.
    """
    print("--- Inizio Addestramento Modello Generale ---")
    
    # 1. Scarica e processa i dati per i soggetti specificati
    all_files = data_loader.fetch_physionet_subjects(subjects=subjects_to_load)
    
    if not all_files:
        print("\nERRORE CRITICO: Nessun file di dati scaricato o trovato.")
        return

    all_features = []
    all_labels = []

    for psg_path, annot_path in all_files:
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is not None and labels is not None:
            features = feature_extractor.extract_psd_features(data)
            all_features.append(features)
            all_labels.append(labels)
        else:
            print(f"Attenzione: Fallito il caricamento dei dati per {os.path.basename(psg_path)}. File saltato.")
            
    if not all_features:
        print("\nERRORE CRITICO: Non è stato possibile estrarre nessuna feature da nessun file.")
        return

    # Concatena i dati di tutti i soggetti
    X_features = np.concatenate(all_features, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    print("\nAddestramento del classificatore classico (Random Forest)...")
    rf_model = models.get_classical_classifier()
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_scaled, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    rf_model.fit(X_train_rf, y_train_rf)
    accuracy = rf_model.score(X_test_rf, y_test_rf)
    print(f"Accuratezza del Random Forest sul set di test: {accuracy:.4f}")
    
    print("\nPreparazione delle sequenze per LSTM...")
    gap = 1 
    X_seq, y_seq = prepare_sequences(X_scaled, y_labels, LOOK_BACK, gap)
    
    if len(X_seq) == 0:
        print("Non ci sono abbastanza dati per creare sequenze con i parametri dati.")
        return

    print(f"Forma delle sequenze di input (X): {X_seq.shape}")
    print(f"Forma delle etichette target (y): {y_seq.shape}")

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

    os.makedirs(models_output_dir, exist_ok=True)
    lstm_model.save(os.path.join(models_output_dir, 'general_lstm_model.h5'))
    joblib.dump(rf_model, os.path.join(models_output_dir, 'general_rf_model.pkl'))
    joblib.dump(scaler, os.path.join(models_output_dir, 'general_scaler.pkl'))
    
    print(f"\nModelli e scaler salvati nella directory '{models_output_dir}'.")


if __name__ == '__main__':
    # Esempio: addestra il modello usando i dati dei primi 10 soggetti
    subjects_to_train_on = list(range(10)) 
    
    train_general_model(subjects_to_load=subjects_to_train_on)
