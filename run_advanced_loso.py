# run_advanced_loso.py

import os
import numpy as np
import pandas as pd

import data_loader
import feature_extractor
import experiment_logic
import utils
from config import (
    OUTPUTS_DIR, UNIQUE_CLASS_NAMES, TRAINING_WINDOWS_MINUTES,
    EPOCH_DURATION, PREDICTION_TARGET_GAP_MINUTES, LOOK_BACK
)

def run_experiment(all_subject_ids):
    print("--- Inizio Esperimento LOSO Avanzato ---")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    final_results = []

    for test_subject_id in all_subject_ids:
        print(f"\n{'='*80}")
        print(f"INIZIO FOLD: SOGGETTO DI TEST = {test_subject_id}")
        print(f"{'='*80}")
        
        train_subject_ids = [s_id for s_id in all_subject_ids if s_id != test_subject_id]
        
        test_files = data_loader.fetch_physionet_subjects(subjects=[test_subject_id])
        if not test_files: continue
        test_data_full, test_labels_full = data_loader.load_sleep_data(test_files[0][0], test_files[0][1])
        if test_data_full is None: continue
        test_features_full = feature_extractor.extract_psd_features(test_data_full)

        train_files = data_loader.fetch_physionet_subjects(subjects=train_subject_ids)
        train_subjects_data = {}
        for i, (psg, annot) in enumerate(train_files):
            subj_id = train_subject_ids[i]
            data, labels = data_loader.load_sleep_data(psg, annot)
            if data is not None:
                features = feature_extractor.extract_psd_features(data)
                train_subjects_data[subj_id] = {'features': features, 'labels': labels}

        for window_min in TRAINING_WINDOWS_MINUTES:
            print(f"\n--- Finestra di Training: {window_min} minuti ---")
            window_epochs = int(window_min * 60 / EPOCH_DURATION)
            
            train_features_window = np.concatenate([d['features'][:window_epochs] for d in train_subjects_data.values() if len(d['features']) >= window_epochs])
            train_labels_window = np.concatenate([d['labels'][:window_epochs] for d in train_subjects_data.values() if len(d['labels']) >= window_epochs])
            
            if len(train_features_window) == 0:
                print(f"Nessun dato di training per la finestra di {window_min} min. Salto.")
                continue

            # --- Task 1: Random Forest ---
            test_features_window = test_features_full[:window_epochs]
            test_labels_window = test_labels_full[:window_epochs]
            if len(test_features_window) > 0:
                rf_results = experiment_logic.train_and_evaluate_rf(
                    train_features_window, train_labels_window,
                    test_features_window, test_labels_window
                )
                final_results.append({
                    'test_subject': test_subject_id, 'model': 'RandomForest',
                    'train_window_min': window_min, 'accuracy': rf_results['accuracy']
                })
                
                # Salva report e matrice di confusione per RF
                fold_dir = os.path.join(OUTPUTS_DIR, f"fold_{test_subject_id}")
                
                title_cm = f"RF - Test Subj {test_subject_id}\nTrain Window: {window_min} min"
                filepath_cm = os.path.join(fold_dir, f"cm_rf_window_{window_min}min.pdf")
                utils.plot_confusion_matrix(rf_results['y_true'], rf_results['y_pred'], UNIQUE_CLASS_NAMES, title_cm, filepath_cm)
                
                filepath_report = os.path.join(fold_dir, f"report_rf_window_{window_min}min.txt")
                with open(filepath_report, 'w') as f:
                    f.write(f"Classification Report for Random Forest\n")
                    f.write(f"Test Subject: {test_subject_id}\n")
                    f.write(f"Training Window: {window_min} minutes\n\n")
                    f.write(rf_results['report'])
                print(f"Report RF salvato in: {filepath_report}")

            # --- Task 2: LSTM ---
            lstm_results, history = experiment_logic.train_and_evaluate_lstm(
                train_features_window, train_labels_window,
                test_features_full, test_labels_full
            )
            
            if lstm_results and history:
                final_results.append({
                    'test_subject': test_subject_id, 'model': 'LSTM',
                    'train_window_min': window_min, 'accuracy': lstm_results['accuracy']
                })
                
                base_path = os.path.join(OUTPUTS_DIR, f"fold_{test_subject_id}", f"lstm_train_window_{window_min}min")
                title_hist = f"LSTM History - Test Subj {test_subject_id}\nTrain Window: {window_min} min"
                utils.plot_history(history, title_hist, f"{base_path}_history.pdf")
                
                title_cm = (f"LSTM - Test Subj {test_subject_id}\nTrain Window: {window_min} min | "
                            f"Predicting > {PREDICTION_TARGET_GAP_MINUTES} min ahead")
                utils.plot_confusion_matrix(lstm_results['y_true'], lstm_results['y_pred'], UNIQUE_CLASS_NAMES, title_cm, f"{base_path}_cm.pdf")

    print(f"\n{'='*80}\n--- ESPERIMENTO COMPLETATO ---\n{'='*80}")
    if not final_results:
        print("Nessun risultato raccolto.")
        return

    results_df = pd.DataFrame(final_results)
    print("\n--- Risultati Complessivi ---")
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, 'advanced_loso_results.csv'), index=False)

    print("\n--- Riepilogo Medie per Configurazione ---")
    summary = results_df.groupby(['model', 'train_window_min'])['accuracy'].agg(['mean', 'std'])
    print(summary)
    summary.to_csv(os.path.join(OUTPUTS_DIR, 'advanced_loso_summary.csv'))
    print(f"\nRisultati salvati nella directory '{OUTPUTS_DIR}'")

if __name__ == '__main__':
    subjects_for_experiment = list(range(3)) 
    # subjects_for_experiment = list(range(83))
    
    run_experiment(all_subject_ids=subjects_for_experiment)