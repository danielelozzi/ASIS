# predict.py

import os
import time
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Importa i nostri moduli personalizzati
import data_loader
import feature_extractor
from config import LOOK_BACK, LABEL_MAP, EPOCH_DURATION

class SleepPredictor:
    """
    Una classe per caricare i modelli addestrati e fare previsioni sugli stadi del sonno.
    """
    def __init__(self, models_dir='models'):
        """
        Inizializza il predictor caricando i modelli e lo scaler.

        Args:
            models_dir (str): La directory dove sono salvati i modelli.
        """
        print("Caricamento dei modelli...")
        try:
            self.lstm_model = load_model(os.path.join(models_dir, 'general_lstm_model.h5'))
            self.rf_model = joblib.load(os.path.join(models_dir, 'general_rf_model.pkl'))
            self.scaler = joblib.load(os.path.join(models_dir, 'general_scaler.pkl'))
            # Crea una mappa inversa per leggere le etichette
            self.inverse_label_map = {v: k for k, v in LABEL_MAP.items()}
            print("Modelli caricati con successo.")
        except IOError as e:
            print(f"Errore: Impossibile caricare i modelli dalla directory '{models_dir}'.")
            print("Assicurati di aver eseguito 'train.py' prima.")
            raise e

    def predict_future_stage_lstm(self, feature_sequence):
        """
        Prevede uno stadio del sonno futuro usando una sequenza di feature.

        Args:
            feature_sequence (np.ndarray): Sequenza di feature con forma 
                                           (1, look_back, n_features).

        Returns:
            tuple: (predicted_stage_int, predicted_stage_str)
        """
        prediction_probs = self.lstm_model.predict(feature_sequence, verbose=0)
        predicted_stage_int = np.argmax(prediction_probs, axis=-1)[0]
        predicted_stage_str = self.inverse_label_map.get(predicted_stage_int, "Sconosciuto")
        return predicted_stage_int, predicted_stage_str

    def classify_stage_rf(self, feature_vector):
        """
        Classifica lo stadio del sonno per un singolo vettore di feature.

        Args:
            feature_vector (np.ndarray): Vettore di feature con forma (1, n_features).

        Returns:
            tuple: (classified_stage_int, classified_stage_str)
        """
        classified_stage_int = self.rf_model.predict(feature_vector)[0]
        classified_stage_str = self.inverse_label_map.get(classified_stage_int, "Sconosciuto")
        return classified_stage_int, classified_stage_str

    def run_simulation(self, psg_path, annot_path, prediction_gap_minutes=60, update_interval_minutes=30):
        """
        Simula la previsione in tempo reale per un singolo soggetto.

        Args:
            psg_path (str): Percorso al file PSG del soggetto.
            annot_path (str): Percorso al file di annotazione del soggetto.
            prediction_gap_minutes (int): Con quanto anticipo fare la previsione (in minuti).
            update_interval_minutes (int): Ogni quanto aggiornare la previsione (in minuti).
        """
        print("\n--- Inizio Simulazione di Previsione Notturna ---")
        
        # 1. Carica e prepara i dati del soggetto
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        if data is None:
            return
            
        features = feature_extractor.extract_psd_features(data)
        scaled_features = self.scaler.transform(features)
        
        # Converti i minuti in numero di epoche
        gap_epochs = int(prediction_gap_minutes * 60 / EPOCH_DURATION)
        update_interval_epochs = int(update_interval_minutes * 60 / EPOCH_DURATION)
        
        total_epochs = len(scaled_features)
        print(f"Soggetto: {os.path.basename(psg_path)}")
        print(f"Durata totale sonno: {total_epochs * EPOCH_DURATION / 3600:.2f} ore")
        print(f"Previsione a {prediction_gap_minutes} min ({gap_epochs} epoche) nel futuro.")
        print(f"Aggiornamento ogni {update_interval_minutes} min ({update_interval_epochs} epoche).\n")

        # 2. Ciclo di simulazione
        for current_epoch in range(LOOK_BACK, total_epochs - gap_epochs, update_interval_epochs):
            current_time_hours = (current_epoch * EPOCH_DURATION) / 3600
            print(f"--- Ora attuale: {current_time_hours:.2f}h dal início del sonno ---")

            # Prepara la sequenza per la previsione LSTM
            start_idx = current_epoch - LOOK_BACK
            end_idx = current_epoch
            sequence = scaled_features[start_idx:end_idx]
            sequence = np.expand_dims(sequence, axis=0) # Rendi (1, look_back, n_features)

            # Prevedi lo stadio futuro con LSTM
            pred_int, pred_str = self.predict_future_stage_lstm(sequence)
            
            # Istante futuro a cui si riferisce la previsione
            future_epoch = current_epoch + gap_epochs
            future_time_hours = (future_epoch * EPOCH_DURATION) / 3600
            
            # Prendi lo stadio reale a quell'istante futuro per confronto
            actual_future_stage_int = labels[future_epoch]
            actual_future_stage_str = self.inverse_label_map.get(actual_future_stage_int, "Sconosciuto")

            print(f"Previsione LSTM per le {future_time_hours:.2f}h: {pred_str} (Reale sarà: {actual_future_stage_str})")

            # Conferma con RF usando i dati fino all'istante attuale
            current_feature_vector = scaled_features[current_epoch].reshape(1, -1)
            rf_int, rf_str = self.classify_stage_rf(current_feature_vector)
            actual_current_stage_str = self.inverse_label_map.get(labels[current_epoch], "Sconosciuto")
            
            print(f"Classificazione RF per l'ora attuale ({current_time_hours:.2f}h): {rf_str} (Reale è: {actual_current_stage_str})\n")
            time.sleep(1) # Pausa per leggibilità

if __name__ == '__main__':
    DATASET_PATH = './sleep-cassette'
    MODELS_DIR = 'models'

    if not os.path.isdir(DATASET_PATH) or not os.path.isdir(MODELS_DIR):
        print("Errore: Assicurati che le directory del dataset e dei modelli esistano.")
    else:
        # Seleziona un soggetto per la simulazione (es. il primo)
        all_files = data_loader.get_subject_files(DATASET_PATH)
        if not all_files:
            print("Nessun soggetto trovato nel dataset.")
        else:
            psg_file, annot_file = all_files[0]
            
            # Crea l'istanza del predictor
            predictor = SleepPredictor(MODELS_DIR)
            
            # Avvia la simulazione
            predictor.run_simulation(
                psg_path=psg_file, 
                annot_path=annot_file,
                prediction_gap_minutes=30,  # Prevedi 30 minuti nel futuro
                update_interval_minutes=15 # Aggiorna ogni 15 minuti
            )
