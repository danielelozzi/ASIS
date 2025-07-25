# run_advanced_loso.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # Per salvare lo scaler

import data_loader
import feature_extractor
import experiment_logic # per prepare_sequences
import models
import utils
from config import (
    OUTPUTS_DIR, UNIQUE_CLASS_NAMES, EPOCH_DURATION, LOOK_BACK,
    TRAIN_SUBJECT_RATIO, RANDOM_SEED, PREDICTION_TARGET_TIMES_MINUTES,
    LSTM_UNITS, LSTM_DROPOUT, LABEL_MAP, FS,
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, LSTM_TRAIN_PREDICTION_GAP_MINUTES,
    MLP_HIDDEN_SIZE, MLP_DROPOUT
)

def run_experiment(all_subject_ids):
    print("--- Inizio Esperimento di Previsione per Soggetto (LSTM Leggero) ---")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    final_results = []

    for subject_id in all_subject_ids:
        print(f"\n{'='*80}")
        print(f"INIZIO VALUTAZIONE PER SOGGETTO: {subject_id}")
        print(f"{'='*80}")
        
        subject_files = data_loader.fetch_physionet_subjects(subjects=[subject_id])
        if not subject_files:
            print(f"Nessun file trovato per il soggetto {subject_id}. Salto.")
            continue
        
        psg_path, annot_path = subject_files[0]
        
        all_data, all_labels = data_loader.load_sleep_data(psg_path, annot_path)
        if all_data is None:
            print(f"Errore caricamento dati per il soggetto {subject_id}. Salto.")
            continue
        
        all_features = feature_extractor.extract_psd_features(all_data)

        training_epochs_count = int(TRAINING_WINDOW_HOURS_PER_SUBJECT * 3600 / EPOCH_DURATION)
        
        if len(all_features) < training_epochs_count + LOOK_BACK + 1: # +1 per almeno un gap
            print(f"Soggetto {subject_id} ha solo {len(all_features)} epoche. Non abbastanza per finestra di {TRAINING_WINDOW_HOURS_PER_SUBJECT} ore + look_back + 1 min di previsione. Salto.")
            continue

        train_features = all_features[:training_epochs_count]
        train_labels = all_labels[:training_epochs_count]
        
        prediction_features_full = all_features[training_epochs_count:]
        prediction_labels_full = all_labels[training_epochs_count:]

        if len(prediction_features_full) == 0:
            print(f"Soggetto {subject_id}: Non ci sono dati sufficienti dopo la finestra di training per la previsione. Salto.")
            continue

        print(f"\n--- Addestramento LSTM sul Soggetto {subject_id} (prime {TRAINING_WINDOW_HOURS_PER_SUBJECT} ore) ---")

        scaler = experiment_logic.StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        
        gap_epochs_for_training = 0
        X_train_seq, y_train_seq = experiment_logic.prepare_sequences(train_scaled, train_labels, LOOK_BACK, gap_epochs_for_training)

        if len(X_train_seq) == 0:
            print(f"Non ci sono abbastanza dati di training per creare sequenze LSTM per il soggetto {subject_id}. Salto.")
            continue

        n_features = X_train_seq.shape[2]
        num_classes = len(np.unique(list(LABEL_MAP.values()))) # Usiamo tutte le possibili classi dalla mappa

        # --- INIZIO: LOGICA PER IL BILANCIAMENTO DELLE CLASSI ---
        class_counts = np.bincount(y_train_seq, minlength=num_classes)
        # Sostituisci 0 con 1 per evitare divisioni per zero se una classe non è presente
        class_counts[class_counts == 0] = 1 
        
        # Calcola i pesi inversamente proporzionali alla frequenza delle classi
        # Minore è la frequenza, maggiore è il peso.
        class_weights = 1. / class_counts
        
        # Mappa i pesi alle etichette di ogni campione nel training set
        sample_weights = class_weights[y_train_seq]
        
        # Crea un sampler pesato
        # Il parametro `num_samples` (opzionale) serve a definire la dimensione dell'epoca.
        # Se non specificato, si userà len(dataset) per un'epoca, ma i batch saranno comunque campionati in modo pesato.
        # Per assicurare che l'epoca abbia la dimensione attesa, lo specifichiamo.
        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y_train_seq), # Campiona quanto il dataset originale
            replacement=True # Necessario per l'oversampling
        )
        # --- FINE: LOGICA PER IL BILANCIAMENTO DELLE CLASSI ---

        lstm_model = models.create_lstm_model(n_features, LSTM_UNITS, 1, num_classes, LSTM_DROPOUT)
        
        dataset = experiment_logic.TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.long))
        
        # Suddividi il dataset PRIMA di applicare il sampler pesato,
        # poiché il sampler pesato si applica al DataLoader
        train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Applica il sampler pesato solo al train_loader
        # Nota: il sampler deve operare sull'intero dataset, non solo sul train_dataset splittato.
        # Per questo, è più robusto creare il sampler PRIMA della split,
        # e poi filtrare i pesi per il train_dataset.
        
        # Un modo più semplice: se il sampler è creato sull'intero dataset,
        # e poi si fa random_split, il sampler deve essere ricreato per train_dataset
        # mappando gli indici.
        # Oppure, si può creare il sampler dopo la split, basandosi solo su train_dataset.
        # Facciamo quest'ultima per semplicità e chiarezza.
        
        # Calcola i pesi solo per il train_dataset effettivo dopo la split
        train_indices = train_dataset.indices
        y_train_seq_actual = y_train_seq[train_indices] # Etichette reali del train_dataset
        
        class_counts_actual = np.bincount(y_train_seq_actual, minlength=num_classes)
        class_counts_actual[class_counts_actual == 0] = 1
        class_weights_actual = 1. / class_counts_actual
        sample_weights_actual = class_weights_actual[y_train_seq_actual]
        
        # Crea il sampler pesato per il train_dataset
        weighted_sampler_train = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights_actual,
            num_samples=len(sample_weights_actual), # Campiona dalla dimensione del training split
            replacement=True
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler_train)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Non mescolare il validation set se non necessario, e senza sampler

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(lstm_model.parameters())
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(EPOCHS):
            lstm_model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for seqs, labs in train_loader:
                optimizer.zero_grad()
                outputs = lstm_model(seqs)
                loss = criterion(outputs, labs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (torch.max(outputs.data, 1)[1] == labs).sum().item()
                train_total += labs.size(0)
            
            lstm_model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for seqs, labs in val_loader:
                    outputs = lstm_model(seqs)
                    val_loss += criterion(outputs, labs).item()
                    val_correct += (torch.max(outputs.data, 1)[1] == labs).sum().item()
                    val_total += labs.size(0)
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_correct / train_total)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_correct / val_total)
            print(f"Epoch {epoch+1}/{EPOCHS} -> Train Acc: {train_correct / train_total:.4f}, Val Acc: {val_correct / val_total:.4f}")

        fold_dir = os.path.join(OUTPUTS_DIR, f"subject_{subject_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        title_hist = f"LSTM Training History - Subject {subject_id}\nTrain Window: {TRAINING_WINDOW_HOURS_PER_SUBJECT} hours"
        utils.plot_history(history, title_hist, os.path.join(fold_dir, f"lstm_training_history_subj_{subject_id}.pdf"))

        print(f"\n--- Valutazione LSTM per Soggetto {subject_id} sui dati rimanenti ---")
        
        test_scaled_features = scaler.transform(prediction_features_full)

        all_y_true_for_cm = []
        all_y_pred_for_cm = []
        
        current_gap_minutes = PREDICTION_INCREMENT_MINUTES 
        
        while True:
            gap_epochs = int(current_gap_minutes * 60 / EPOCH_DURATION)
            
            X_test_seq, y_test_true = experiment_logic.prepare_sequences(
                test_scaled_features, prediction_labels_full, LOOK_BACK, gap_epochs
            )

            # Controlla se ci sono dati sufficienti per le previsioni
            # per evitare loop infiniti o errori di indice.
            # Se X_test_seq è vuoto, significa che non ci sono più sequenze valide
            # per il gap corrente e i successivi.
            if len(X_test_seq) == 0:
                # Se il gap_epochs è già così grande che non ci sono sequenze
                # nel test set, allora abbiamo finito.
                # Per la prima iterazione (current_gap_minutes = PREDICTION_INCREMENT_MINUTES),
                # se non ci sono dati, è un problema.
                # Per le iterazioni successive, significa che abbiamo esaurito i dati.
                if current_gap_minutes == PREDICTION_INCREMENT_MINUTES and len(prediction_features_full) > 0:
                    print(f"  - Gap di {current_gap_minutes} min: Non abbastanza dati nel set di previsione per creare sequenze con look_back e gap. (Dati previsione disponibili: {len(prediction_features_full)} epoche). Salto.")
                elif len(prediction_features_full) == 0:
                     print(f"  - Nessun dato di previsione disponibile dopo la finestra di training. Fine previsioni.")
                else:
                    print(f"  - Raggiunto fine dati per previsioni. Ultimo gap tentato: {current_gap_minutes - PREDICTION_INCREMENT_MINUTES} min. Stop previsioni.")
                break # Esci dal ciclo while

            with torch.no_grad():
                outputs = lstm_model(torch.tensor(X_test_seq, dtype=torch.float32))
                y_test_pred = torch.max(outputs, 1)[1].numpy()
            
            accuracy = experiment_logic.accuracy_score(y_test_true, y_test_pred)
            print(f"  - Soggetto {subject_id} - Previsione a {TRAINING_WINDOW_HOURS_PER_SUBJECT}h e {current_gap_minutes}min (dall'inizio): Accuratezza = {accuracy:.4f}")
            
            final_results.append({
                'subject_id': subject_id,
                'model': 'LSTM_SubjectSpecific',
                'train_window_hours': TRAINING_WINDOW_HOURS_PER_SUBJECT,
                'prediction_gap_minutes': current_gap_minutes,
                'accuracy': accuracy,
                'num_predictions': len(y_test_true)
            })
            
            all_y_true_for_cm.extend(y_test_true)
            all_y_pred_for_cm.extend(y_test_pred)
            
            current_gap_minutes += PREDICTION_INCREMENT_MINUTES
        
        if all_y_true_for_cm:
            title_cm_overall = (f"LSTM - Subject {subject_id}\n"
                                f"Train Window: {TRAINING_WINDOW_HOURS_PER_SUBJECT} hours\n"
                                f"Aggregated Predictions from {PREDICTION_INCREMENT_MINUTES} min onwards")
            
            filepath_cm_overall = os.path.join(fold_dir, f"cm_lstm_aggregated_subj_{subject_id}.pdf")
            utils.plot_confusion_matrix(all_y_true_for_cm, all_y_pred_for_cm, UNIQUE_CLASS_NAMES, title_cm_overall, filepath_cm_overall)
            print(f"Matrice di confusione aggregata per il soggetto {subject_id} salvata in: {filepath_cm_overall}")
        else:
            print(f"Nessuna previsione effettuata per il soggetto {subject_id} per generare la matrice di confusione aggregata.")


    print(f"\n{'='*80}\n--- ESPERIMENTO COMPLETATO ---\n{'='*80}")
    if not final_results:
        print("Nessun risultato raccolto.")
        return

    results_df = pd.DataFrame(final_results)
    print("\n--- Risultati Complessivi ---")
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, 'subject_specific_lstm_results.csv'), index=False)

    print("\n--- Riepilogo Medie per Gap di Previsione ---")
    # Aggiungi 'count' per vedere quanti soggetti hanno contribuito a quel gap
    summary = results_df.groupby(['prediction_gap_minutes'])['accuracy'].agg(['mean', 'std', 'count'])
    print(summary)
    summary.to_csv(os.path.join(OUTPUTS_DIR, 'subject_specific_lstm_summary_by_gap.csv'))
    
    print(f"\nRisultati salvati nella directory '{OUTPUTS_DIR}'")

if __name__ == '__main__':
    subjects_for_experiment = list(range(3))
    
    run_experiment(all_subject_ids=subjects_for_experiment)