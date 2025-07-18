import os
import train
import predict
import pandas as pd
import utils
from config import (
    OUTPUTS_DIR, UNIQUE_CLASS_NAMES,
    TRAIN_PREDICTION_GAP_MINUTES, LOOK_BACK, EPOCH_DURATION # <-- CORREZIONE QUI
)

def run_loso_experiment(all_subject_ids):
    print("--- Inizio Esperimento Leave-One-Subject-Out ---")
    print(f"Soggetti totali da processare: {all_subject_ids}")
    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    all_fold_results = []

    for test_subject_id in all_subject_ids:
        print(f"\n{'='*60}")
        print(f"INIZIO FOLD: Soggetto di Test = {test_subject_id}")
        print(f"{'='*60}")
        
        train_subject_ids = [s_id for s_id in all_subject_ids if s_id != test_subject_id]
        fold_output_dir = os.path.join(OUTPUTS_DIR, f'fold_test_subject_{test_subject_id}')
        
        print("\n--- Fase di Addestramento ---")
        train.train_on_subjects(
            subjects_to_load=train_subject_ids,
            models_output_dir=fold_output_dir
        )
        
        print("\n--- Fase di Valutazione ---")
        eval_results = predict.evaluate_on_subject(
            subject_id=test_subject_id,
            models_dir=fold_output_dir
        )
        
        if not eval_results:
            print(f"Valutazione fallita per il soggetto {test_subject_id}. Salto il fold.")
            continue

        # Salva le matrici di confusione e raccogli i risultati
        current_fold_summary = {'test_subject': test_subject_id}
        
        # Matrice di confusione per Random Forest
        if 'rf' in eval_results and eval_results['rf']:
            rf_res = eval_results['rf']
            current_fold_summary['rf_accuracy'] = rf_res['accuracy']
            title = f"CM for Random Forest (Instantaneous)\nTest Subject: {test_subject_id}"
            filepath = os.path.join(fold_output_dir, 'confusion_matrix_rf.pdf')
            utils.plot_confusion_matrix(rf_res['y_true'], rf_res['y_pred'], UNIQUE_CLASS_NAMES, title, filepath)

        # Matrici di confusione per LSTM
        if 'lstm' in eval_results and eval_results['lstm']:
            for gap_min, lstm_res in eval_results['lstm'].items():
                current_fold_summary[f'lstm_accuracy_{gap_min}min'] = lstm_res['accuracy']
                title = (f"CM for LSTM - Prediction at {gap_min} min\n"
                         f"Test Subject: {test_subject_id} | Trained with {TRAIN_PREDICTION_GAP_MINUTES} min gap | "
                         f"Input: {LOOK_BACK * EPOCH_DURATION / 60:.1f} min of sleep")
                filepath = os.path.join(fold_output_dir, f'confusion_matrix_lstm_{gap_min}min.pdf')
                utils.plot_confusion_matrix(lstm_res['y_true'], lstm_res['y_pred'], UNIQUE_CLASS_NAMES, title, filepath)

        all_fold_results.append(current_fold_summary)
        print(f"\nFINE FOLD: Soggetto di Test = {test_subject_id}")

    print(f"\n{'='*60}")
    print("--- Esperimento LOSO Completato ---")
    print(f"{'='*60}")

    if not all_fold_results:
        print("Nessun risultato è stato raccolto.")
        return

    results_df = pd.DataFrame(all_fold_results).set_index('test_subject')
    mean_results = results_df.mean()
    std_results = results_df.std()
    summary_df = pd.DataFrame({'Media': mean_results, 'Dev. Std.': std_results})
    
    print("\n--- Riepilogo Risultati (per Fold) ---")
    print(results_df)
    print("\n\n--- Statistiche Complessive (Media e Dev. Std.) ---")
    print(summary_df)

    results_df.to_csv(os.path.join(OUTPUTS_DIR, 'loso_results_per_fold.csv'))
    summary_df.to_csv(os.path.join(OUTPUTS_DIR, 'loso_results_summary.csv'))
    print(f"\nI risultati sono stati salvati nella directory '{OUTPUTS_DIR}'.")


if __name__ == '__main__':
    # ATTENZIONE: Eseguire l'esperimento completo richiede MOLTO tempo.
    # Inizia con un piccolo sottoinsieme di soggetti per testare il codice.
    subjects_for_experiment = list(range(3)) # Esempio con 3 soggetti
    
    # Per eseguire l'esperimento completo (83 soggetti), decommenta la linea seguente:
    # subjects_for_experiment = list(range(83))
    
    run_loso_experiment(all_subject_ids=subjects_for_experiment)
