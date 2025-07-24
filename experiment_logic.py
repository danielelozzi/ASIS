# experiment_logic.py

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import models
from config import (
    LOOK_BACK, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, LSTM_UNITS,
    LSTM_DROPOUT, EPOCH_DURATION, UNIQUE_CLASS_NAMES,
    LABEL_MAP
)

def prepare_sequences(features, labels, look_back, gap_epochs):
    X, y = [], []
    # Assicurati che ci sia abbastanza spazio per la sequenza e il gap
    for i in range(len(features) - look_back - gap_epochs):
        X.append(features[i:i + look_back])
        y.append(labels[i + look_back + gap_epochs])
    return np.array(X), np.array(y)

def train_and_evaluate_rf(train_features, train_labels, test_features, test_labels):
    """Addestra e valuta un modello Random Forest, restituendo anche il report."""
    print("Addestramento e valutazione Random Forest (non usato in questo esperimento)...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    # Chiama il modello che ora è bilanciato di default
    model = models.get_classical_classifier()
    model.fit(train_scaled, train_labels)
    
    predictions = model.predict(test_scaled)
    accuracy = accuracy_score(test_labels, predictions)
    
    # Genera il report di classificazione
    all_possible_labels = sorted(list(set(LABEL_MAP.values())))
    report = classification_report(
        test_labels, 
        predictions, 
        target_names=UNIQUE_CLASS_NAMES,
        labels=all_possible_labels,
        zero_division=0
    )
    
    print(f"RF Accuracy: {accuracy:.4f}")
    print("RF Classification Report:\n", report)
    
    return {
        'accuracy': accuracy, 
        'y_true': test_labels, 
        'y_pred': predictions,
        'report': report # Aggiunto il report ai risultati
    }

# La funzione train_and_evaluate_lstm non è più utilizzata direttamente
# nel nuovo flusso di run_advanced_loso.py.
# La lasciamo qui, ma la sua logica è stata assorbita e adattata nel file principale.
def train_and_evaluate_lstm(train_features, train_labels, test_features, test_labels):
    """
    Addestra e valuta un modello LSTM.
    Questa funzione non è più utilizzata direttamente nel nuovo flusso di training per soggetto.
    """
    print("`train_and_evaluate_lstm` non è più utilizzata nel flusso principale.")
    # Puoi lasciare il codice esistente o rimuoverlo se sei sicuro che non ti servirà più.
    # Per semplicità, la rendo un placeholder che indica che non è usata.
    return None, None