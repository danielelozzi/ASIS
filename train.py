# train.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    LSTM_UNITS,
    LSTM_DROPOUT
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
    """
    print("--- Inizio Addestramento Modello Generale (PyTorch) ---")
    
    # 1. Scarica e processa i dati
    all_files = data_loader.fetch_physionet_subjects(subjects=subjects_to_load)
    
    if not all_files:
        print("\nERRORE CRITICO: Nessun file di dati scaricato o trovato.")
        return

    all_features, all_labels = [], []
    for psg_path, annot_path in all_files:
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is not None and labels is not None:
            features = feature_extractor.extract_psd_features(data)
            all_features.append(features)
            all_labels.append(labels)
    
    if not all_features:
        print("\nERRORE CRITICO: Non è stato possibile estrarre nessuna feature.")
        return

    X_features = np.concatenate(all_features, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # 2. Addestra il modello Random Forest
    print("\nAddestramento del classificatore classico (Random Forest)...")
    rf_model = models.get_classical_classifier()
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_scaled, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    rf_model.fit(X_train_rf, y_train_rf)
    accuracy = rf_model.score(X_test_rf, y_test_rf)
    print(f"Accuratezza del Random Forest: {accuracy:.4f}")
    
    # 3. Prepara i dati per LSTM
    print("\nPreparazione delle sequenze per LSTM...")
    gap = 1 
    X_seq, y_seq = prepare_sequences(X_scaled, y_labels, LOOK_BACK, gap)
    
    if len(X_seq) == 0:
        print("Non ci sono abbastanza dati per creare sequenze.")
        return

    # 4. Addestra il modello LSTM con PyTorch
    n_features = X_seq.shape[2]
    num_classes = len(np.unique(y_labels))

    lstm_model = models.create_lstm_model(
        input_size=n_features,
        hidden_size=LSTM_UNITS,
        num_layers=1,
        num_classes=num_classes,
        dropout=LSTM_DROPOUT
    )
    print(lstm_model)

    # Converti i dati in tensori PyTorch
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.long)

    # Crea dataset e dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int((1.0 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Definisci loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters())

    print("\nAddestramento del modello LSTM con PyTorch...")
    for epoch in range(EPOCHS):
        lstm_model.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validazione
        lstm_model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = lstm_model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_acc += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # 5. Salva i modelli
    os.makedirs(models_output_dir, exist_ok=True)
    torch.save(lstm_model.state_dict(), os.path.join(models_output_dir, 'general_lstm_model.pth'))
    joblib.dump(rf_model, os.path.join(models_output_dir, 'general_rf_model.pkl'))
    joblib.dump(scaler, os.path.join(models_output_dir, 'general_scaler.pkl'))
    
    print(f"\nModelli e scaler salvati nella directory '{models_output_dir}'.")


if __name__ == '__main__':
    subjects_to_train_on = list(range(10)) 
    train_general_model(subjects_to_load=subjects_to_train_on)