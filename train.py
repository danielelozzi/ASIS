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

import data_loader
import feature_extractor
import models
from config import (
    LOOK_BACK, 
    BATCH_SIZE, 
    EPOCHS, 
    VALIDATION_SPLIT, 
    LSTM_UNITS,
    LSTM_DROPOUT
)

def prepare_sequences(features, labels, look_back, gap):
    X, y = [], []
    for i in range(len(features) - look_back - gap):
        X.append(features[i:i + look_back])
        y.append(labels[i + look_back + gap])
    return np.array(X), np.array(y)

def train_general_model(subjects_to_load, models_output_dir='models'):
    print("--- Inizio Addestramento Modello Generale (PyTorch) ---")
    
    all_files = data_loader.fetch_physionet_subjects(subjects=subjects_to_load)
    if not all_files:
        print("\nERRORE: Nessun file di dati trovato.")
        return

    all_features, all_labels = [], []
    for psg_path, annot_path in all_files:
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is not None:
            features = feature_extractor.extract_psd_features(data)
            all_features.append(features)
            all_labels.append(labels)
    
    X_features = np.concatenate(all_features)
    y_labels = np.concatenate(all_labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    print("\nAddestramento Random Forest...")
    rf_model = models.get_classical_classifier()
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_scaled, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    rf_model.fit(X_train_rf, y_train_rf)
    print(f"Accuratezza Random Forest: {rf_model.score(X_test_rf, y_test_rf):.4f}")
    
    print("\nPreparazione sequenze per LSTM...")
    X_seq, y_seq = prepare_sequences(X_scaled, y_labels, LOOK_BACK, gap=1)
    
    n_features = X_seq.shape[2]
    num_classes = len(np.unique(y_labels))
    lstm_model = models.create_lstm_model(
        input_size=n_features,
        hidden_size=LSTM_UNITS,
        num_layers=1,
        num_classes=num_classes,
        dropout=LSTM_DROPOUT
    )
    
    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.long))
    train_size = int((1.0 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters())
    
    print("\nAddestramento LSTM con PyTorch...")
    for epoch in range(EPOCHS):
        lstm_model.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        lstm_model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = lstm_model(sequences)
                val_loss += criterion(outputs, labels).item()
                val_acc += ((torch.max(outputs.data, 1)[1]) == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc/len(val_dataset):.4f}")

    os.makedirs(models_output_dir, exist_ok=True)
    torch.save(lstm_model.state_dict(), os.path.join(models_output_dir, 'general_lstm_model.pth'))
    joblib.dump(rf_model, os.path.join(models_output_dir, 'general_rf_model.pkl'))
    joblib.dump(scaler, os.path.join(models_output_dir, 'general_scaler.pkl'))
    print(f"\nModelli salvati in '{models_output_dir}'.")

if __name__ == '__main__':
    train_general_model(subjects_to_load=list(range(10)))