# predict.py

import os
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import data_loader
import feature_extractor
import models
from train import prepare_sequences # Importiamo la funzione per preparare le sequenze
from config import (
    LOOK_BACK, 
    LABEL_MAP, 
    EPOCH_DURATION, 
    LSTM_UNITS, 
    LSTM_DROPOUT, 
    BANDS, 
    EEG_CHANNELS,
    EVALUATION_GAPS_MINUTES
)

class SleepEvaluator:
    def __init__(self, models_dir):
        print(f"Caricamento modelli da: {models_dir}")
        try:
            self.scaler = joblib.load(os.path.join(models_dir, 'loso_scaler.pkl'))
            self.rf_model = joblib.load(os.path.join(models_dir, 'loso_rf_model.pkl'))

            n_features = len(BANDS) * len(EEG_CHANNELS)
            num_classes = len(LABEL_MAP)
            
            self.lstm_model = models.create_lstm_model(
                input_size=n_features,
                hidden_size=LSTM_UNITS,
                num_layers=1,
                num_classes=num_classes,
                dropout=LSTM_DROPOUT
            )
            self.lstm_model.load_state_dict(torch.load(os.path.join(models_dir, 'loso_lstm_model.pth')))
            self.lstm_model.eval()

            self.inverse_label_map = {v: k for k, v in LABEL_MAP.items()}
            print("Modelli caricati.")
        except IOError as e:
            raise Exception(f"Errore caricamento modelli da '{models_dir}'.") from e

    def evaluate_subject(self, psg_path, annot_path):
        """
        Valuta le performance dei modelli su un soggetto specifico per diversi gap temporali.
        """
        print(f"\n--- Valutazione sul soggetto: {os.path.basename(psg_path)} ---")
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is None: return {}

        features = feature_extractor.extract_psd_features(data)
        scaled_features = self.scaler.transform(features)
        
        results = {}

        # Valutazione Random Forest (classificazione istantanea)
        y_pred_rf = self.rf_model.predict(scaled_features)
        rf_accuracy = accuracy_score(labels, y_pred_rf)
        results['rf_accuracy_current_epoch'] = rf_accuracy
        print(f"Accuratezza Random Forest (epoca corrente): {rf_accuracy:.4f}")

        # Valutazione LSTM per ogni gap
        print("\nValutazione LSTM su diversi gap temporali:")
        for gap_min in EVALUATION_GAPS_MINUTES:
            gap_epochs = int(gap_min * 60 / EPOCH_DURATION)
            
            X_seq, y_true = prepare_sequences(scaled_features, labels, LOOK_BACK, gap_epochs)
            
            if len(X_seq) == 0:
                print(f"  - Gap di {gap_min} min: Non ci sono abbastanza dati per la valutazione. Salto.")
                continue

            sequence_tensor = torch.tensor(X_seq, dtype=torch.float32)
            
            with torch.no_grad():
                outputs = self.lstm_model(sequence_tensor)
                # Calcola le probabilità con Softmax
                probabilities = F.softmax(outputs, dim=1).numpy()
                # Prendi la classe predetta
                y_pred = torch.max(outputs, 1)[1].numpy()

            lstm_accuracy = accuracy_score(y_true, y_pred)
            
            # Calcola la probabilità media di essere sveglio ('W', etichetta 0)
            avg_wake_prob = np.mean(probabilities[:, 0])

            key_acc = f'lstm_accuracy_{gap_min}min'
            key_wake_prob = f'lstm_avg_wake_prob_{gap_min}min'
            
            results[key_acc] = lstm_accuracy
            results[key_wake_prob] = avg_wake_prob

            print(f"  - Gap di {gap_min} min: Accuratezza LSTM = {lstm_accuracy:.4f} | Prob. media risveglio = {avg_wake_prob:.4f}")
            
        return results


# NUOVA FUNZIONE per essere chiamata dallo script principale
def evaluate_on_subject(subject_id, models_dir):
    """
    Funzione di alto livello per caricare un soggetto e valutarlo.
    """
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

if __name__ == '__main__':
    print("Questo script è pensato per essere importato, non eseguito direttamente.")
    print("Usa 'run_loso_experiment.py' per avviare la valutazione.")