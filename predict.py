import os
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import data_loader
import feature_extractor
import models
from train import prepare_sequences
from config import (
    LOOK_BACK, LABEL_MAP, EPOCH_DURATION, LSTM_UNITS, LSTM_DROPOUT, 
    BANDS, EEG_CHANNELS, EVALUATION_GAPS_MINUTES
)

class SleepEvaluator:
    def __init__(self, models_dir):
        print(f"Caricamento modelli da: {models_dir}")
        try:
            self.scaler = joblib.load(os.path.join(models_dir, 'loso_scaler.pkl'))
            self.rf_model = joblib.load(os.path.join(models_dir, 'loso_rf_model.pkl'))

            n_features = len(BANDS) * len(EEG_CHANNELS)
            num_classes = len(np.unique(list(LABEL_MAP.values())))
            
            self.lstm_model = models.create_lstm_model(
                input_size=n_features, hidden_size=LSTM_UNITS,
                num_layers=1, num_classes=num_classes, dropout=LSTM_DROPOUT
            )
            self.lstm_model.load_state_dict(torch.load(os.path.join(models_dir, 'loso_lstm_model.pth')))
            self.lstm_model.eval()
            print("Modelli caricati.")
        except IOError as e:
            raise Exception(f"Errore caricamento modelli da '{models_dir}'.") from e

    def evaluate_subject(self, psg_path, annot_path):
        print(f"\n--- Valutazione sul soggetto: {os.path.basename(psg_path)} ---")
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is None: return {}

        features = feature_extractor.extract_psd_features(data)
        scaled_features = self.scaler.transform(features)
        
        results = {'rf': {}, 'lstm': {}}

        # Valutazione Random Forest
        y_pred_rf = self.rf_model.predict(scaled_features)
        results['rf']['accuracy'] = accuracy_score(labels, y_pred_rf)
        results['rf']['y_true'] = labels
        results['rf']['y_pred'] = y_pred_rf
        print(f"Accuratezza Random Forest (epoca corrente): {results['rf']['accuracy']:.4f}")

        # Valutazione LSTM per ogni gap
        print("\nValutazione LSTM su diversi gap temporali:")
        for gap_min in EVALUATION_GAPS_MINUTES:
            gap_epochs = int(gap_min * 60 / EPOCH_DURATION)
            X_seq, y_true_lstm = prepare_sequences(scaled_features, labels, LOOK_BACK, gap_epochs)
            
            if len(X_seq) == 0:
                print(f"  - Gap di {gap_min} min: Non ci sono abbastanza dati. Salto.")
                continue

            sequence_tensor = torch.tensor(X_seq, dtype=torch.float32)
            with torch.no_grad():
                outputs = self.lstm_model(sequence_tensor)
                y_pred_lstm = torch.max(outputs, 1)[1].numpy()

            acc = accuracy_score(y_true_lstm, y_pred_lstm)
            results['lstm'][gap_min] = {
                'accuracy': acc,
                'y_true': y_true_lstm,
                'y_pred': y_pred_lstm
            }
            print(f"  - Gap di {gap_min} min: Accuratezza LSTM = {acc:.4f}")
            
        return results

def evaluate_on_subject(subject_id, models_dir):
    try:
        evaluator = SleepEvaluator(models_dir=models_dir)
        subject_files = data_loader.fetch_physionet_subjects(subjects=[subject_id])
        if subject_files:
            psg, annot = subject_files[0]
            results = evaluator.evaluate_subject(psg, annot)
            return results
        else:
            print(f"Nessun file trovato per il soggetto {subject_id}")
            return None
    except Exception as e:
        print(f"\nValutazione fallita per il soggetto {subject_id}: {e}")
        return None