# run_loso_experiment.py

import os
import train
import predict
import pandas as pd

def run_loso_experiment(all_subject_ids):
    """
    Esegue un esperimento di cross-validation Leave-One-Subject-Out.
    
    Per ogni soggetto:
    1. Lo imposta come soggetto di test.
    2. Addestra i modelli su TUTTI gli altri soggetti.
    3. Valuta le performance sul soggetto di test.
    """
    
    print("--- Inizio Esperimento Leave-One-Subject-Out ---")
    print(f"Soggetti totali da processare: {all_subject_ids}")
    
    all_results = []

    for test_subject_id in all_subject_ids:
        print(f"\n{'='*60}")
        print(f"INIZIO FOLD: Soggetto di Test = {test_subject_id}")
        print(f"{'='*60}")
        
        # 1. Definisci i soggetti per il training
        train_subject_ids = [s_id for s_id in all_subject_ids if s_id != test_subject_id]
        
        # 2. Definisci una directory unica per i modelli di questo fold
        models_dir_for_fold = os.path.join('models', f'loso_fold_test_subject_{test_subject_id}')
        
        # 3. Addestra i modelli sui dati di training
        print("\n--- Fase di Addestramento ---")
        train.train_on_subjects(
            subjects_to_load=train_subject_ids,
            models_output_dir=models_dir_for_fold
        )
        
        # 4. Valuta i modelli sul soggetto di test
        print("\n--- Fase di Valutazione ---")
        fold_results = predict.evaluate_on_subject(
            subject_id=test_subject_id,
            models_dir=models_dir_for_fold
        )
        
        if fold_results:
            fold_results['test_subject'] = test_subject_id
            all_results.append(fold_results)
        
        print(f"\nFINE FOLD: Soggetto di Test = {test_subject_id}")

    print(f"\n{'='*60}")
    print("--- Esperimento LOSO Completato ---")
    print(f"{'='*60}")

    if not all_results:
        print("Nessun risultato è stato raccolto. Controllare gli errori nei log.")
        return

    # 5. Mostra i risultati finali in un formato leggibile
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index('test_subject')
    
    # Calcola le medie e le deviazioni standard
    mean_results = results_df.mean()
    std_results = results_df.std()
    
    print("\n--- Riepilogo Risultati (per Fold) ---")
    print(results_df)
    
    print("\n\n--- Statistiche Complessive (Media e Dev. Std.) ---")
    summary_df = pd.DataFrame({'Media': mean_results, 'Dev. Std.': std_results})
    print(summary_df)

    # Salva i risultati in un file CSV per analisi future
    results_df.to_csv('loso_results_per_fold.csv')
    summary_df.to_csv('loso_results_summary.csv')
    print("\nI risultati sono stati salvati in 'loso_results_per_fold.csv' e 'loso_results_summary.csv'")


if __name__ == '__main__':
    # ATTENZIONE: Il dataset PhysioNet Sleep-EDF ha 83 soggetti (da 0 a 82).
    # Eseguire l'esperimento completo richiede MOLTO tempo.
    # Inizia con un piccolo sottoinsieme di soggetti per testare il codice.
    
    # Esempio con 4 soggetti (0, 1, 2, 3)
    subjects_for_experiment = list(range(4)) 
    
    # Per eseguire l'esperimento completo, decommenta la linea seguente:
    # subjects_for_experiment = list(range(83))
    
    run_loso_experiment(all_subject_ids=subjects_for_experiment)