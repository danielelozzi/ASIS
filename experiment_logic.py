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
    LSTM_DROPOUT, PREDICTION_TARGET_GAP_MINUTES, EPOCH_DURATION, UNIQUE_CLASS_NAMES
)

def prepare_sequences(features, labels, look_back, gap_epochs):
    X, y = [], []
    for i in range(len(features) - look_back - gap_epochs):
        X.append(features[i:i + look_back])
        y.append(labels[i + look_back + gap_epochs])
    return np.array(X), np.array(y)

def train_and_evaluate_rf(train_features, train_labels, test_features, test_labels):
    """Addestra e valuta un modello Random Forest, restituendo anche il report."""
    print("Addestramento e valutazione Random Forest...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    # Chiama il modello che ora è bilanciato di default
    model = models.get_classical_classifier()
    model.fit(train_scaled, train_labels)
    
    predictions = model.predict(test_scaled)
    accuracy = accuracy_score(test_labels, predictions)
    
    # Genera il report di classificazione
    report = classification_report(
        test_labels, 
        predictions, 
        target_names=UNIQUE_CLASS_NAMES,
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

def train_and_evaluate_lstm(train_features, train_labels, test_features, test_labels):
    """Addestra e valuta un modello LSTM."""
    print("Addestramento e valutazione LSTM...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    
    gap_epochs = int(PREDICTION_TARGET_GAP_MINUTES * 60 / EPOCH_DURATION)
    X_train_seq, y_train_seq = prepare_sequences(train_scaled, train_labels, LOOK_BACK, gap_epochs)
    
    if len(X_train_seq) == 0:
        print("Non ci sono abbastanza dati di training per creare sequenze LSTM.")
        return None, None

    n_features = X_train_seq.shape[2]
    num_classes = len(np.unique(train_labels))
    model = models.create_lstm_model(n_features, LSTM_UNITS, 1, num_classes, LSTM_DROPOUT)
    
    dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.long))
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for seqs, labs in train_loader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (torch.max(outputs.data, 1)[1] == labs).sum().item()
            train_total += labs.size(0)
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for seqs, labs in val_loader:
                outputs = model(seqs)
                val_loss += criterion(outputs, labs).item()
                val_correct += (torch.max(outputs.data, 1)[1] == labs).sum().item()
                val_total += labs.size(0)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Val Acc: {val_correct / val_total:.4f}")

    test_scaled = scaler.transform(test_features)
    X_test_seq, y_test_true = prepare_sequences(test_scaled, test_labels, LOOK_BACK, gap_epochs)

    if len(X_test_seq) == 0:
        print("Non ci sono abbastanza dati di test per la valutazione LSTM.")
        return None, history

    with torch.no_grad():
        outputs = model(torch.tensor(X_test_seq, dtype=torch.float32))
        y_test_pred = torch.max(outputs, 1)[1].numpy()
        
    accuracy = accuracy_score(y_test_true, y_test_pred)
    print(f"LSTM Accuracy: {accuracy:.4f}")
    
    results = {'accuracy': accuracy, 'y_true': y_test_true, 'y_pred': y_test_pred}
    return results, history