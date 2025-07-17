# predict.py

import os
import time
import numpy as np
import joblib
import torch

# Importa i nostri moduli personalizzati
import data_loader
import feature_extractor
import models
from config import LOOK_BACK, LABEL_MAP, EPOCH_DURATION, LSTM_UNITS, LSTM_DROPOUT

class SleepPredictor:
    """
    Carica modelli addestrati (PyTorch e Scikit-learn) per fare previsioni.
    """
    def __init__(self, models_dir='models'):
        print("Caricamento dei modelli...")
        try:
            # Carica scaler e Random Forest
            self.scaler = joblib.load(os.path.join(models_dir, 'general_scaler.pkl'))
            self.rf_model = joblib.load(os.path.join(models_dir, 'general_rf_model.pkl'))

            # Carica il modello LSTM PyTorch
            # Assumiamo che il numero di feature e classi sia derivabile,
            # ma è meglio essere espliciti se possibile.
            # Qui usiamo dei valori fissi basati sulla configurazione
            n_features = len(feature_extractor.BANDS) * len(data_loader.EEG_CHANNELS)
            num_classes = len(LABEL_MAP)
            
            self.lstm_model = models.create_lstm_model(
                input_size=n_features,
                hidden_size=LSTM_UNITS,
                num_layers=1,
                num_classes=num_classes,
                dropout=LSTM_DROPOUT
            )
            self.lstm_model.load_state_dict(torch.load(os.path.join(models_dir, 'general_lstm_model.pth')))
            self.lstm_model.eval() # Imposta il modello in modalità valutazione

            self.inverse_label_map = {v: k for k, v in LABEL_MAP.items()}
            print("Modelli caricati con successo.")
        except IOError as e:
            print(f"Errore: Impossibile caricare i modelli da '{models_dir}'.")
            print("Assicurati di aver eseguito 'train.py' prima.")
            raise e

    def predict_future_stage_lstm(self, feature_sequence):
        """
        Prevede uno stadio del sonno futuro con LSTM (PyTorch).
        """
        # Converte la sequenza in un tensore PyTorch
        sequence_tensor = torch.tensor(feature_sequence, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.lstm_model(sequence_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        predicted_stage_int = predicted.item()
        predicted_stage_str = self.inverse_label_map.get(predicted_stage_int, "Sconosciuto")
        return predicted_stage_int, predicted_stage_str

    def classify_stage_rf(self, feature_vector):
        """
        Classifica lo stadio del sonno corrente con Random Forest.
        """
        classified_stage_int = self.rf_model.predict(feature_vector)[0]
        classified_stage_str = self.inverse_label_map.get(classified_stage_int, "Sconosciuto")
        return classified_stage_int, classified_stage_str

    def run_simulation(self, psg_path, annot_path, prediction_gap_minutes=60, update_interval_minutes=30):
        print("\n--- Inizio Simulazione di Previsione Notturna ---")
        
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is None: return
            
        features = feature_extractor.extract_psd_features(data)
        scaled_features = self.scaler.transform(features)
        
        gap_epochs = int(prediction_gap_minutes * 60 / EPOCH_DURATION)
        update_interval_epochs = int(update_interval_minutes * 60 / EPOCH_DURATION)
        total_epochs = len(scaled_features)
        
        print(f"Soggetto: {os.path.basename(psg_path)}")
        print(f"Previsione a {prediction_gap_minutes} min nel futuro.")
        
        for current_epoch in range(LOOK_BACK, total_epochs - gap_epochs, update_interval_epochs):
            current_time_hours = (current_epoch * EPOCH_DURATION) / 3600
            print(f"\n--- Ora attuale: {current_time_hours:.2f}h ---")

            start_idx = current_epoch - LOOK_BACK
            sequence = scaled_features[start_idx:current_epoch]
            sequence = np.expand_dims(sequence, axis=0)

            pred_int, pred_str = self.predict_future_stage_lstm(sequence)
            
            future_epoch = current_epoch + gap_epochs
            future_time_hours = (future_epoch * EPOCH_DURATION) / 3600
            actual_future_stage_str = self.inverse_label_map.get(labels[future_epoch], "Sconosciuto")

            print(f"Previsione LSTM per le {future_time_hours:.2f}h: {pred_str} (Reale sarà: {actual_future_stage_str})")

            current_feature_vector = scaled_features[current_epoch].reshape(1, -1)
            rf_int, rf_str = self.classify_stage_rf(current_feature_vector)
            actual_current_stage_str = self.inverse_label_map.get(labels[current_epoch], "Sconosciuto")
            
            print(f"Classificazione RF attuale: {rf_str} (Reale è: {actual_current_stage_str})")
            time.sleep(1)

if __name__ == '__main__':
    # Esempio: esegue una simulazione su un soggetto
    # Prima scarica i dati per un soggetto di test
    subjects_to_test = [10] # Usa un soggetto diverso da quelli di training
    all_files = data_loader.fetch_physionet_subjects(subjects=subjects_to_test)

    if not all_files:
        print("Nessun file scaricato per la simulazione.")
    else:
        psg_file, annot_file = all_files[0]
        
        try:
            predictor = SleepPredictor(models_dir='models')
            predictor.run_simulation(
                psg_path=psg_file, 
                annot_path=annot_file,
                prediction_gap_minutes=30,
                update_interval_minutes=15
            )
        except Exception as e:
            print(f"\nSimulazione fallita: {e}")