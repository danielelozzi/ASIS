import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import data_loader
import feature_extractor
import models
import utils
from config import (
    LOOK_BACK, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, LSTM_UNITS,
    LSTM_DROPOUT, TRAIN_PREDICTION_GAP_MINUTES, EPOCH_DURATION
)

def prepare_sequences(features, labels, look_back, gap_epochs):
    """ Prepara sequenze per l'LSTM. """
    X, y = [], []
    for i in range(len(features) - look_back - gap_epochs):
        X.append(features[i:i + look_back])
        y.append(labels[i + look_back + gap_epochs])
    return np.array(X), np.array(y)

def train_on_subjects(subjects_to_load, models_output_dir):
    """
    Addestra i modelli sui soggetti specificati e salva i risultati.
    """
    print(f"--- Inizio Addestramento per {len(subjects_to_load)} soggetti ---")
    print(f"--- I modelli verranno salvati in: {models_output_dir} ---")

    all_files = data_loader.fetch_physionet_subjects(subjects=subjects_to_load)
    if not all_files:
        print("\nERRORE: Nessun file di dati trovato.")
        return None

    all_features, all_labels = [], []
    for psg_path, annot_path in all_files:
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is not None:
            features = feature_extractor.extract_psd_features(data)
            all_features.append(features)
            all_labels.append(labels)
    
    if not all_features:
        print("\nERRORE: Nessuna feature estratta. Impossibile continuare.")
        return None

    X_features = np.concatenate(all_features)
    y_labels = np.concatenate(all_labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    print("\nAddestramento Random Forest...")
    rf_model = models.get_classical_classifier()
    rf_model.fit(X_scaled, y_labels)
    print("Addestramento Random Forest completato.")
    
    print("\nPreparazione sequenze per LSTM...")
    gap_epochs = int(TRAIN_PREDICTION_GAP_MINUTES * 60 / EPOCH_DURATION)
    X_seq, y_seq = prepare_sequences(X_scaled, y_labels, LOOK_BACK, gap_epochs)
    
    n_features = X_seq.shape[2]
    num_classes = len(np.unique(y_labels))
    
    lstm_model = models.create_lstm_model(
        input_size=n_features, hidden_size=LSTM_UNITS,
        num_layers=1, num_classes=num_classes, dropout=LSTM_DROPOUT
    )
    
    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.long))
    train_size = int((1.0 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters())
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nAddestramento LSTM con PyTorch (Gap di {TRAIN_PREDICTION_GAP_MINUTES} min)...")
    for epoch in range(EPOCHS):
        lstm_model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            train_total += labels.size(0)
        
        lstm_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = lstm_model(sequences)
                val_loss += criterion(outputs, labels).item()
                val_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                val_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    os.makedirs(models_output_dir, exist_ok=True)
    torch.save(lstm_model.state_dict(), os.path.join(models_output_dir, 'loso_lstm_model.pth'))
    joblib.dump(rf_model, os.path.join(models_output_dir, 'loso_rf_model.pkl'))
    joblib.dump(scaler, os.path.join(models_output_dir, 'loso_scaler.pkl'))
    print(f"\nModelli per questo fold salvati in '{models_output_dir}'.")
    
    # Salva la cronologia e il grafico
    utils.save_history(history, os.path.join(models_output_dir, 'training_history.json'))
    utils.plot_history(history, os.path.join(models_output_dir, 'training_history.pdf'))

    return history