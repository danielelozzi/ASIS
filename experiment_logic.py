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
    MLP_HIDDEN_SIZE, MLP_DROPOUT # Nuovi import per i parametri MLP
)

def run_experiment(all_subject_ids):
    print("--- Inizio Esperimento di Predizione Temporale (Modello Generale con Feature Temporale) ---")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # --- 1. Suddivisione dei Soggetti in Training e Test ---
    np.random.seed(RANDOM_SEED) # Imposta il seed per la riproducibilità di shuffle
    shuffled_subject_ids = np.random.permutation(all_subject_ids) # Permuta per assicurare split casuale
    
    num_train_subjects = int(len(shuffled_subject_ids) * TRAIN_SUBJECT_RATIO)
    train_subject_ids = sorted(shuffled_subject_ids[:num_train_subjects].tolist())
    test_subject_ids = sorted(shuffled_subject_ids[num_train_subjects:].tolist())
    
    print(f"\nSoggetti per Training ({len(train_subject_ids)}): {train_subject_ids}")
    print(f"Soggetti per Test ({len(test_subject_ids)}): {test_subject_ids}")

    # --- 2. Caricamento e Pre-elaborazione Dati di Training (prime 3 ore per soggetto) ---
    print("\n--- Caricamento e preparazione dati di Training (80% dei soggetti, prime 3 ore) ---")
    
    # La finestra di training fissa per il modello generale (prime 3 ore)
    training_hours_for_model = 3
    training_epochs_count = int(training_hours_for_model * 3600 / EPOCH_DURATION)

    all_train_features = []
    all_train_labels = []

    for s_id in train_subject_ids:
        subject_files = data_loader.fetch_physionet_subjects(subjects=[s_id])
        if not subject_files:
            print(f"Nessun file trovato per il soggetto di training {s_id}. Salto.")
            continue
        
        psg_path, annot_path = subject_files[0]
        data, labels = data_loader.load_sleep_data(psg_path, annot_path)
        
        if data is None:
            print(f"Errore caricamento dati per il soggetto di training {s_id}. Salto.")
            continue
        
        if len(data) >= training_epochs_count + LOOK_BACK + int(LSTM_TRAIN_PREDICTION_GAP_MINUTES * 60 / EPOCH_DURATION):
            features = feature_extractor.extract_psd_features(data[:training_epochs_count])
            all_train_features.append(features)
            all_train_labels.append(labels[:training_epochs_count])
        else:
            print(f"Soggetto di training {s_id} ha solo {len(data)} epoche, non sufficienti per {training_hours_for_model}h + look_back + {LSTM_TRAIN_PREDICTION_GAP_MINUTES}min. Salto.")
    
    if not all_train_features:
        print("\nERRORE: Nessun dato di training valido è stato caricato. Impossibile continuare.")
        return

    X_train_full = np.concatenate(all_train_features)
    y_train_full = np.concatenate(all_train_labels)

    # Scalatura dei dati di training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    
    # Salva lo scaler per usarlo nella fase di test
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, 'scaler_general_model.pkl'))
    print(f"Scaler salvato in: {os.path.join(OUTPUTS_DIR, 'scaler_general_model.pkl')}")

    # Preparazione sequenze per LSTM con time_feature
    # Per il training, il time_feature riflette LSTM_TRAIN_PREDICTION_GAP_MINUTES
    # Max duration in minutes per la normalizzazione del time_feature (es. 8 ore)
    max_night_duration_minutes = 8 * 60
    
    X_train_seq, y_train_seq, train_time_features = experiment_logic.prepare_sequences(
        X_train_scaled, y_train_full, LOOK_BACK, LSTM_TRAIN_PREDICTION_GAP_MINUTES, max_night_duration_minutes
    )

    if len(X_train_seq) == 0:
        print("Non ci sono abbastanza dati di training per creare sequenze LSTM. Fine.")
        return

    n_features = X_train_seq.shape[2]
    num_classes = len(np.unique(list(LABEL_MAP.values())))

    # --- Bilanciamento delle Classi per il Training LSTM ---
    class_counts_train = np.bincount(y_train_seq, minlength=num_classes)
    class_counts_train[class_counts_train == 0] = 1
    
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(train_time_features, dtype=torch.float32), # Aggiungi time_feature qui
        torch.tensor(y_train_seq, dtype=torch.long)
    )
    
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_indices = train_dataset.indices
    y_train_seq_actual = y_train_seq[train_indices]
    
    class_counts_actual = np.bincount(y_train_seq_actual, minlength=num_classes)
    class_counts_actual[class_counts_actual == 0] = 1
    sample_weights_actual = (1. / class_counts_actual)[y_train_seq_actual]
    
    weighted_sampler_train = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights_actual,
        num_samples=len(sample_weights_actual),
        replacement=True
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler_train)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. Addestramento del Modello LSTM Generale ---
    print("\n--- Addestramento del Modello LSTM Generale sui dati di Training ---")
    lstm_model_general = models.create_lstm_temporal_predictor(
        input_size=n_features,
        hidden_size=LSTM_UNITS,
        num_layers=1, # Num_layers impostato a 1 come richiesto per "leggero"
        num_classes=num_classes,
        lstm_dropout=LSTM_DROPOUT,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        mlp_dropout=MLP_DROPOUT
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model_general.parameters())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        lstm_model_general.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for seqs, time_feats, labs in train_loader: # Ricevi anche time_features
            optimizer.zero_grad()
            outputs = lstm_model_general(seqs, time_feats) # Passa entrambi gli input
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (torch.max(outputs.data, 1)[1] == labs).sum().item()
            train_total += labs.size(0)
        
        lstm_model_general.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for seqs, time_feats, labs in val_loader: # Ricevi anche time_features
                outputs = lstm_model_general(seqs, time_feats) # Passa entrambi gli input
                val_loss += criterion(outputs, labs).item()
                val_correct += (torch.max(outputs.data, 1)[1] == labs).sum().item()
                val_total += labs.size(0)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Acc: {train_correct / train_total:.4f}, Val Acc: {val_correct / val_total:.4f}")

    # Salva il modello generale
    model_path = os.path.join(OUTPUTS_DIR, 'lstm_general_model.pth')
    torch.save(lstm_model_general.state_dict(), model_path)
    print(f"\nModello generale salvato in '{model_path}'.")
    
    # Salva la cronologia e il grafico del training del modello generale
    title_hist_general = f"LSTM General Model Training History\n(Trained on {len(train_subject_ids)} subjects)"
    utils.plot_history(history, title_hist_general, os.path.join(OUTPUTS_DIR, 'lstm_general_training_history.pdf'))

    # --- 4. Valutazione sui Soggetti di Test (20%) ---
    print("\n--- Inizio Valutazione sui Soggetti di Test (20%) ---")
    
    test_results = []
    
    # Ricarica lo scaler per sicurezza (o usalo direttamente se non l'hai scaricato)
    loaded_scaler = joblib.load(os.path.join(OUTPUTS_DIR, 'scaler_general_model.pkl'))

    # Carica il modello addestrato
    loaded_model = models.create_lstm_temporal_predictor(
        input_size=n_features,
        hidden_size=LSTM_UNITS,
        num_layers=1,
        num_classes=num_classes,
        lstm_dropout=LSTM_DROPOUT,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        mlp_dropout=MLP_DROPOUT
    )
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval() # Imposta il modello in modalità valutazione

    for s_id in test_subject_ids:
        print(f"\nValutazione sul soggetto di test: {s_id}")
        subject_files = data_loader.fetch_physionet_subjects(subjects=[s_id])
        if not subject_files:
            print(f"Nessun file trovato per il soggetto di test {s_id}. Salto.")
            continue
        
        psg_path, annot_path = subject_files[0]
        all_data, all_labels = data_loader.load_sleep_data(psg_path, annot_path)
        
        if all_data is None:
            print(f"Errore caricamento dati per il soggetto di test {s_id}. Salto.")
            continue
        
        all_features = feature_extractor.extract_psd_features(all_data)
        all_features_scaled = loaded_scaler.transform(all_features) # Scala con lo scaler addestrato

        # Dati di partenza per la sequenza LSTM: le prime 3 ore
        # L'ultima epoca utile per formare una sequenza che predica oltre le 3 ore.
        # Una sequenza LSTM di LOOK_BACK epoche che finisce all'epoca (3h - 1)
        # inizia all'epoca (3h - LOOK_BACK).
        # Quindi, la sequenza è da `(training_epochs_count - LOOK_BACK)` a `(training_epochs_count - 1)`.

        base_training_end_epoch = training_epochs_count - 1 # L'ultima epoca inclusa nella "finestra di training"
        
        # Per predire un target a `prediction_target_time_minutes` (es. 3.5 ore = 210 min)
        # L'epoca corrispondente al target è `target_epoch = int(prediction_target_time_minutes * 60 / EPOCH_DURATION)`
        
        # La sequenza LSTM che predice `target_epoch` deve finire a `target_epoch - time_gap_epochs`.
        # E iniziare a `target_epoch - time_gap_epochs - LOOK_BACK`.
        
        # Ridefiniamo la logica: Vogliamo prevedere *a* un certo `target_time_minutes` dall'inizio del sonno.
        # La sequenza di input dell'LSTM sarà sempre le `LOOK_BACK` epoche che precedono quel `target_time_minutes`
        # meno il `LSTM_TRAIN_PREDICTION_GAP_MINUTES` che il modello è stato addestrato a colmare.
        
        # Quindi, se vogliamo prevedere all'epoca X:
        # Il target_epoch_idx è int(target_time_minutes * 60 / EPOCH_DURATION)
        
        # La sequenza LSTM deve finire a (target_epoch_idx - lstm_train_gap_epochs - 1)
        # E iniziare a (target_epoch_idx - lstm_train_gap_epochs - LOOK_BACK)
        
        # La time_feature da dare al modello sarà target_time_minutes / max_night_duration_minutes

        # Inizializza liste per tutte le previsioni del soggetto di test
        all_y_true_subject = []
        all_y_pred_subject = []

        for target_time_min in PREDICTION_TARGET_TIMES_MINUTES:
            target_epoch_idx = int(target_time_min * 60 / EPOCH_DURATION)
            
            # Calcola l'indice di inizio della sequenza e l'indice di fine della sequenza.
            # La sequenza LSTM deve essere "LOOK_BACK" epoche prima del target
            # E "lstm_train_gap_epochs" epoche prima del target
            # in base a come è stato addestrato il modello.

            # L'ultima epoca della sequenza LOOK_BACK (idx_end_seq_inclusive)
            idx_end_seq_inclusive = target_epoch_idx - lstm_train_gap_epochs - 1
            
            # L'inizio della sequenza (idx_start_seq)
            idx_start_seq = idx_end_seq_inclusive - LOOK_BACK + 1

            # Controlli sui limiti dei dati
            if idx_start_seq < 0 or target_epoch_idx >= len(all_features_scaled):
                print(f"  - Soggetto {s_id}: Non abbastanza dati per prevedere a {target_time_min} min (Necessari fino all'epoca {target_epoch_idx}). Salto.")
                continue

            # Estrai la sequenza per la previsione
            current_X_seq = all_features_scaled[idx_start_seq : idx_end_seq_inclusive + 1] # +1 perché slice è esclusiva
            current_y_true = all_labels[target_epoch_idx]
            
            # La feature temporale per questa previsione
            current_time_feature = torch.tensor([[target_time_min / max_night_duration_minutes]], dtype=torch.float32)

            # Rimodella la sequenza per il modello (batch_size=1)
            current_X_seq_tensor = torch.tensor(current_X_seq, dtype=torch.float32).unsqueeze(0) # Aggiunge dimensione batch

            # Effettua la previsione
            with torch.no_grad():
                output = loaded_model(current_X_seq_tensor, current_time_feature)
                predicted_label = torch.max(output, 1)[1].item()

            test_results.append({
                'subject_id': s_id,
                'model': 'LSTM_General',
                'prediction_target_time_min': target_time_min,
                'true_label': current_y_true,
                'predicted_label': predicted_label,
                'is_correct': (current_y_true == predicted_label)
            })
            
            all_y_true_subject.append(current_y_true)
            all_y_pred_subject.append(predicted_label)
            
            print(f"  - Previsione per {target_time_min} min: Vero={current_y_true}, Predetto={predicted_label} (Corretto: {current_y_true == predicted_label})")

        # Genera una matrice di confusione aggregata per tutte le previsioni di questo soggetto
        if all_y_true_subject:
            subject_test_dir = os.path.join(OUTPUTS_DIR, f"test_subject_{s_id}")
            os.makedirs(subject_test_dir, exist_ok=True)

            title_cm_subject = (f"LSTM General Model - Test Subject {s_id}\n"
                                f"Aggregated Predictions from {PREDICTION_TARGET_TIMES_MINUTES[0]} min onwards")
            filepath_cm_subject = os.path.join(subject_test_dir, f"cm_lstm_general_subj_{s_id}.pdf")
            utils.plot_confusion_matrix(all_y_true_subject, all_y_pred_subject, UNIQUE_CLASS_NAMES, title_cm_subject, filepath_cm_subject)
            print(f"Matrice di confusione aggregata per il soggetto {s_id} salvata in: {filepath_cm_subject}")
        else:
            print(f"Nessuna previsione valida effettuata per il soggetto {s_id} per generare la matrice di confusione aggregata.")


    print(f"\n{'='*80}\n--- ESPERIMENTO COMPLETATO ---\n{'='*80}")
    if not test_results:
        print("Nessun risultato di test raccolto.")
        return

    results_df = pd.DataFrame(test_results)
    print("\n--- Risultati Complessivi Dettagliati (Test Set) ---")
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, 'general_model_test_results_detailed.csv'), index=False)

    print("\n--- Riepilogo Accuratezza Media per Gap di Previsione (Test Set) ---")
    summary = results_df.groupby('prediction_target_time_min')['is_correct'].agg(['mean', 'std', 'count']).rename(columns={'mean': 'accuracy'})
    print(summary)
    summary.to_csv(os.path.join(OUTPUTS_DIR, 'general_model_test_summary_by_gap.csv'))
    
    print(f"\nRisultati salvati nella directory '{OUTPUTS_DIR}'")

if __name__ == '__main__':
    all_subjects = list(range(83)) # Tutti i soggetti disponibili
    # all_subjects = list(range(10)) # Per un test più rapido

    run_experiment(all_subject_ids=all_subjects)