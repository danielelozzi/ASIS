# predict.py

import os
import time
import numpy as np
import joblib
import torch

import data_loader
import feature_extractor
import models
from config import LOOK_BACK, LABEL_MAP, EPOCH_DURATION, LSTM_UNITS, LSTM_DROPOUT, BANDS, EEG_CHANNELS

class SleepPredictor:
    def __init__(self, models_dir='models'):
        print("Caricamento modelli...")
        try:
            self.scaler = joblib.load(os.path.join(models_dir, 'general_scaler.pkl'))
            self.rf_model = joblib.load(os.path.join(models_dir, 'general_rf_model.pkl'))

            n_features = len(BANDS) * len(EEG_CHANNELS)
            num_classes = len(LABEL_MAP)
            
            self.lstm_model = models.create_lstm_model(
                input_size=n_features,
                hidden_size=LSTM_UNITS,
                num_layers=1,
                num_classes=num_classes,
                dropout=LSTM_DROPOUT
            )
            self.lstm_model.load_state_dict(torch.load(os.path.join(models_dir, 'general_lstm_model.pth')))
            self.lstm_model.eval()

            self.inverse_label_map = {v: k for k, v in LABEL_MAP.items()}
            print("Modelli caricati.")
        except IOError as e:
            raise Exception(f"Errore caricamento modelli da '{models_dir}'. Eseguire prima 'train.py'.") from e

    def predict_future_stage_lstm(self, feature_sequence):
        sequence_tensor = torch.tensor(feature_sequence, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.lstm_model(sequence_tensor)
            predicted_int = torch.max(outputs.data, 1)[1].item()
        return predicted_int, self.inverse_label_map.get(predicted_int, "Sconosciuto")

    def classify_stage_rf(self, feature_vector):
        predicted_int = self.rf_model.predict(feature_vector)[0]
        return predicted_int, self.inverse_label_map.get(predicted_int, "Sconosciuto")

    def run_simulation(self, psg_path, annot_path, prediction_gap_minutes=30, update_interval_minutes=15):
        print("\n--- Inizio Simulazione ---")
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is None: return
            
        features = feature_extractor.extract_psd_features(data)
        scaled_features = self.scaler.transform(features)
        
        gap_epochs = int(prediction_gap_minutes * 60 / EPOCH_DURATION)
        update_interval_epochs = int(update_interval_minutes * 60 / EPOCH_DURATION)
        
        for current_epoch in range(LOOK_BACK, len(scaled_features) - gap_epochs, update_interval_epochs):
            current_time_h = (current_epoch * EPOCH_DURATION) / 3600
            print(f"\n--- Ora: {current_time_h:.2f}h ---")

            sequence = np.expand_dims(scaled_features[current_epoch - LOOK_BACK:current_epoch], axis=0)
            pred_int, pred_str = self.predict_future_stage_lstm(sequence)
            
            future_epoch = current_epoch + gap_epochs
            future_time_h = (future_epoch * EPOCH_DURATION) / 3600
            actual_future_str = self.inverse_label_map.get(labels[future_epoch], "Sconosciuto")
            print(f"Previsione LSTM per le {future_time_h:.2f}h: {pred_str} (Reale: {actual_future_str})")

            rf_int, rf_str = self.classify_stage_rf(scaled_features[current_epoch].reshape(1, -1))
            actual_current_str = self.inverse_label_map.get(labels[current_epoch], "Sconosciuto")
            print(f"Classificazione RF attuale: {rf_str} (Reale: {actual_current_str})")
            time.sleep(1)

if __name__ == '__main__':
    try:
        predictor = SleepPredictor(models_dir='models')
        # Usa un soggetto non visto durante il training per la simulazione
        test_subject_files = data_loader.fetch_physionet_subjects(subjects=[10])
        if test_subject_files:
            psg, annot = test_subject_files[0]
            predictor.run_simulation(psg, annot)
    except Exception as e:
        print(f"\nSimulazione fallita: {e}")