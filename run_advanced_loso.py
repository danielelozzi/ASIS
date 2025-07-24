# run_advanced_loso.py

import os
import numpy as np
import pandas as pd
import torch # Necessario per il caricamento del modello LSTM

import data_loader
import feature_extractor
import experiment_logic # Useremo la sua funzione train_and_evaluate_lstm
import models # Per caricare la struttura del modello LSTM
import utils
from config import (
    OUTPUTS_DIR, UNIQUE_CLASS_NAMES, EPOCH_DURATION, LOOK_BACK,
    TRAINING_WINDOW_HOURS_PER_SUBJECT, PREDICTION_INCREMENT_MINUTES,
    LSTM_UNITS, LSTM_DROPOUT, LABEL_MAP, FS
)

def run_experiment(all_subject_ids):
    print("--- Inizio Esperimento di Previsione per Soggetto (LSTM Leggero) ---")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    final_results = []

    for subject_id in all_subject_ids:
        print(f"\n{'='*80}")
        print(f"INIZIO VALUTAZIONE PER SOGGETTO: {subject_id}")
        print(f"{'='*80}")
        
        # --- Caricamento e preparazione dati del soggetto corrente ---
        subject_files = data_loader.fetch_physionet_subjects(subjects=[subject_id])
        if not subject_files:
            print(f"Nessun file trovato per il soggetto {subject_id}. Salto.")
            continue
        
        psg_path, annot_path = subject_files[0]
        
        # Carica tutti i dati del soggetto
        all_data, all_labels = data_loader.load_sleep_data(psg_path, annot_path)
        if all_data is None:
            print(f"Errore caricamento dati per il soggetto {subject_id}. Salto.")
            continue
        
        all_features = feature_extractor.extract_psd_features(all_data)

        # Definisci la finestra di training iniziale (le prime 3 ore)
        training_epochs_count = int(TRAINING_WINDOW_HOURS_PER_SUBJECT * 3600 / EPOCH_DURATION)
        
        if len(all_features) < training_epochs_count + LOOK_BACK: # + LOOK_BACK per la sequenza LSTM minima
            print(f"Soggetto {subject_id} ha solo {len(all_features)} epoche. Non abbastanza per finestra di {TRAINING_WINDOW_HOURS_PER_SUBJECT} ore + look_back. Salto.")
            continue

        train_features = all_features[:training_epochs_count]
        train_labels = all_labels[:training_epochs_count]
        
        # Dati per la previsione (dal termine della finestra di training fino alla fine)
        prediction_features_full = all_features[training_epochs_count:]
        prediction_labels_full = all_labels[training_epochs_count:]

        if len(prediction_features_full) == 0:
            print(f"Soggetto {subject_id}: Non ci sono dati sufficienti dopo la finestra di training per la previsione. Salto.")
            continue

        print(f"\n--- Addestramento LSTM sul Soggetto {subject_id} (prime {TRAINING_WINDOW_HOURS_PER_SUBJECT} ore) ---")

        # Non utilizziamo più experiment_logic.train_and_evaluate_lstm direttamente
        # per il testing, ma il training è interno a questo script.
        # Useremo un processo di training simile a quello di train.py ma qui.

        # Scalatura dei dati di training
        scaler = experiment_logic.StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        
        # Preparazione sequenze per LSTM (per il training)
        # Nota: il gap qui è 0 perché stiamo addestrando sulla finestra di training
        # e vogliamo che il target sia l'epoca immediatamente successiva all'ultima epoca di look_back.
        # Il vero "gap" di previsione sarà applicato durante la valutazione.
        gap_epochs_for_training = 0
        X_train_seq, y_train_seq = experiment_logic.prepare_sequences(train_scaled, train_labels, LOOK_BACK, gap_epochs_for_training)

        if len(X_train_seq) == 0:
            print(f"Non ci sono abbastanza dati di training per creare sequenze LSTM per il soggetto {subject_id}. Salto.")
            continue

        n_features = X_train_seq.shape[2]
        num_classes = len(np.unique(train_labels)) # Numero di classi presenti nel set di training

        # Inizializza e addestra il modello LSTM
        lstm_model = models.create_lstm_model(n_features, LSTM_UNITS, 1, num_classes, LSTM_DROPOUT)
        
        dataset = experiment_logic.TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.long))
        train_size = int((1 - experiment_logic.VALIDATION_SPLIT) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = experiment_logic.DataLoader(train_dataset, batch_size=experiment_logic.BATCH_SIZE, shuffle=True)
        val_loader = experiment_logic.DataLoader(val_dataset, batch_size=experiment_logic.BATCH_SIZE)
        
        criterion = experiment_logic.nn.CrossEntropyLoss()
        optimizer = experiment_logic.optim.Adam(lstm_model.parameters())
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(experiment_logic.EPOCHS):
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
            print(f"Epoch {epoch+1}/{experiment_logic.EPOCHS} -> Val Acc: {val_correct / val_total:.4f}")

        fold_dir = os.path.join(OUTPUTS_DIR, f"subject_{subject_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Salva la cronologia e il grafico del training
        title_hist = f"LSTM Training History - Subject {subject_id}\nTrain Window: {TRAINING_WINDOW_HOURS_PER_SUBJECT} hours"
        utils.plot_history(history, title_hist, os.path.join(fold_dir, f"lstm_training_history_subj_{subject_id}.pdf"))

        print(f"\n--- Valutazione LSTM per Soggetto {subject_id} sui dati rimanenti ---")
        
        # Scalatura dei dati di previsione
        prediction_scaled_features_full = scaler.transform(prediction_features_full)

        # Inizializza un dizionario per raccogliere tutti i veri e previsti per la matrice di confusione aggregata
        all_y_true_for_cm = []
        all_y_pred_for_cm = []

        # Genera i gap di previsione dinamicamente
        # Il massimo gap è la lunghezza dei dati di previsione meno LOOK_BACK,
        # poiché non possiamo prevedere oltre i dati disponibili.
        max_prediction_epochs = len(prediction_features_full) - LOOK_BACK
        
        if max_prediction_epochs <= 0:
            print(f"Non abbastanza dati per la previsione con look_back per il soggetto {subject_id}.")
        else:
            # Converti max_prediction_epochs in minuti
            max_prediction_minutes = (max_prediction_epochs * EPOCH_DURATION) // 60

            # Vogliamo iniziare la previsione da 3h e 0 min + PREDICTION_INCREMENT_MINUTES
            # Questo significa che il target è a gap_min minuti dopo la fine della finestra di training.
            # Il primo gap effettivo sarà 3h e X min, dove X è PREDICTION_INCREMENT_MINUTES.
            # L'indice iniziale da cui iniziare la previsione in prediction_features_full è 0.

            # I gap_minutes si riferiscono al tempo *dall'inizio* della parte di previsione.
            # Esempio: se PREDICTION_INCREMENT_MINUTES = 1, il primo target è a 1 minuto
            # dall'inizio della parte di previsione.
            
            # Genera i gap in minuti
            # Consideriamo il PREDICTION_INCREMENT_MINUTES come il più piccolo "salto" in avanti
            # dopo la fine del look_back dalla finestra di training.
            # Il loop continuerà finché c'è abbastanza futuro per formare una sequenza e un target.
            
            # Calcola l'indice di inizio per la previsione relativa ai dati di `prediction_features_full`
            # La prima previsione sarà per l'epoca (LOOK_BACK + gap_epochs_first_target).
            
            # Inizia a prevedere da 1 minuto dopo la fine del training window + look_back
            # fino alla fine dei dati disponibili.
            
            # Il gap target è relativo all'ultimo punto della finestra di LOOK_BACK
            # Quindi, se vogliamo prevedere X minuti avanti, il target sarà
            # l'epoca corrente + LOOK_BACK + (X minuti convertiti in epoche).
            
            # Qui stiamo prevedendo in avanti dal punto in cui il training window finisce.
            # Il LOOK_BACK è già incluso nella `prepare_sequences`.
            # Quello che vogliamo è prevedere `PREDICTION_INCREMENT_MINUTES` in avanti.
            
            # La logica è:
            # Prendiamo una finestra di LOOK_BACK epoche.
            # Prevediamo per l'epoca che si trova a `gap_epochs_to_predict` *dopo* la fine di quella finestra di LOOK_BACK.
            
            # Per questo esperimento, vogliamo prevedere *dopo* le 3 ore di training.
            # Quindi l'indice `i` in `prepare_sequences` sarà l'inizio della sequenza.
            # Il target sarà `i + LOOK_BACK + gap_epochs`.
            
            # Per predire 3h e 30m, 3h e 31m, ..., 3h e Nmin:
            # La finestra di training finisce a `training_epochs_count`.
            # La prima sequenza per la previsione inizierà a `training_epochs_count - LOOK_BACK`.
            # Il suo target sarà `training_epochs_count + gap_epochs`.

            # Modifichiamo la preparazione delle sequenze per la previsione.
            # Vogliamo prendere sequenze dalla parte di `all_features` che è *dopo* il training set.
            # Quindi `prediction_scaled_features_full` e `prediction_labels_full` sono i dati di test.
            
            # Il `gap_epochs` in `prepare_sequences` è il gap tra la fine della sequenza LOOK_BACK e il target.
            
            # Per "3 ore e mezza", il gap è 30 minuti (dopo le 3 ore).
            # Per "3 ore e 31 minuti", il gap è 31 minuti.
            
            # Vogliamo iterare dal primo minuto dopo la fine delle 3 ore di training
            # fino alla fine dei dati disponibili, con passi di 1 minuto.
            
            current_gap_minutes = PREDICTION_INCREMENT_MINUTES # Inizia a prevedere X minuti dopo la fine della finestra di training
            
            while True:
                gap_epochs = int(current_gap_minutes * 60 / EPOCH_DURATION)
                
                # prepare_sequences prende le features e labels del "test set" (qui, prediction_scaled_features_full)
                # Il target è (current_index + LOOK_BACK + gap_epochs) all'interno di prediction_scaled_features_full.
                
                # Consideriamo il "tempo 0" come l'inizio della `prediction_scaled_features_full`.
                # Quindi, una sequenza di `LOOK_BACK` epoche inizia a `i`.
                # Il suo target è a `i + LOOK_BACK + gap_epochs`.
                
                # Se `i + LOOK_BACK + gap_epochs` supera la lunghezza di `prediction_scaled_features_full`,
                # non ci sono più dati per questa previsione.
                
                # Quindi, le sequenze devono essere create con un "offset" dall'inizio della parte di previsione.
                # Per prevedere 1 minuto dopo la fine delle 3 ore (cioè 3h e 1m):
                # La sequenza dovrebbe finire all'epoca 3h. Il target è 3h + 1m.
                # La nostra `prepare_sequences` prende una finestra di `look_back` e un target `gap_epochs` dopo.
                
                # Per semplificare, useremo `prepare_sequences` su `all_features` ma modificheremo gli indici.
                
                # L'idea è: addestriamo sul soggetto_id[:training_epochs_count].
                # Poi valutiamo sul soggetto_id[training_epochs_count:]
                # Vogliamo predire l'epoca X minuti dopo l'epoca Y, dove Y è l'ultima epoca *conosciuta*.
                # L'ultima epoca conosciuta è training_epochs_count - 1.
                
                # Per prevedere 3h e 30min, 3h e 31min, ecc.
                # L'ultimo dato di training è `training_epochs_count - 1`.
                # Il primo dato che vogliamo prevedere è all'epoca `training_epochs_count + gap_epochs_relative_to_end_of_training`.
                
                # La `prepare_sequences` funziona così:
                # X.append(features[i:i + look_back])
                # y.append(labels[i + look_back + gap_epochs])
                
                # Per il training, `features` e `labels` sono `train_scaled` e `train_labels`.
                # `gap_epochs_for_training` è 0. Quindi `y` è `labels[i + look_back]`. Previsione immediata.
                
                # Per la previsione dopo le 3 ore:
                # `features` e `labels` sono `all_features_scaled` e `all_labels`.
                # `i` deve iniziare dalla posizione in cui possiamo formare una sequenza.
                # e il target deve essere `training_epochs_count + (current_gap_minutes * 60 / EPOCH_DURATION)`.
                
                # Quindi, le sequenze per la previsione devono iniziare *prima* del punto di previsione.
                # Se il target è l'epoca `T_target`, la sequenza LSTM deve finire a `T_target - gap_epochs`.
                # E la sequenza deve iniziare a `T_target - gap_epochs - LOOK_BACK`.
                
                # Costruiamo i dati di test per ogni gap dinamicamente.
                # Il `test_start_index` è l'indice in `all_features` da cui iniziamo a considerare le sequenze per la previsione.
                # Questo indice deve essere tale che `test_start_index + LOOK_BACK + gap_epochs` non superi la lunghezza totale dei dati.
                
                # La sequenza deve finire *almeno* all'indice `training_epochs_count - 1`.
                # Quindi la prima sequenza utile per prevedere oltre il training window
                # dovrebbe iniziare a `training_epochs_count - LOOK_BACK`.
                
                # L'indice del *target* nel dataset completo `all_labels`
                target_epoch_global_idx = training_epochs_count + gap_epochs
                
                # Se il target è oltre la fine dei dati disponibili, smettiamo.
                if target_epoch_global_idx >= len(all_labels):
                    # print(f"Raggiunto fine dati per gap di {current_gap_minutes} min. Stop previsioni.")
                    break
                
                # La sequenza che predice questo target deve finire a `target_epoch_global_idx - gap_epochs`.
                # E iniziare a `target_epoch_global_idx - gap_epochs - LOOK_BACK`.
                
                # Questa logica di creazione di sequenze e target per la fase di previsione è cruciale.
                # `prepare_sequences(features, labels, look_back, gap_epochs)`
                # i-esima sequenza: features[i:i + look_back]
                # i-esimo target: labels[i + look_back + gap_epochs]
                
                # Vogliamo che la prima previsione sia per 3 ore e 1 minuto.
                # L'indice del target è (training_epochs_count) + (1 minuto in epoche).
                # Quindi, l'indice `i` per la sequenza deve essere:
                # i + LOOK_BACK + gap_epochs = target_epoch_global_idx
                # i = target_epoch_global_idx - LOOK_BACK - gap_epochs
                
                # Questo `i` deve essere >= 0.
                
                # Dobbiamo creare le sequenze per la previsione che terminano prima del target di previsione.
                # E i target devono essere specifici per ogni "gap".
                
                # Questo è un approccio comune:
                # X_test, y_true_test = [], []
                # for i in range(len(all_features) - LOOK_BACK - gap_epochs):
                #     if (i + LOOK_BACK + gap_epochs) >= training_epochs_count: # Inizia a creare sequenze solo dopo il training window
                #         X_test.append(all_features[i:i + LOOK_BACK])
                #         y_true_test.append(all_labels[i + LOOK_BACK + gap_epochs])
                
                # Questo loop può essere inefficiente. Un modo migliore è identificare gli indici.
                
                # L'epoca di inizio della sequenza nel dataset COMPLETO `all_features`
                start_seq_global_idx = training_epochs_count - LOOK_BACK # L'ultima sequenza di training utile che finisce a training_epochs_count-1
                
                # L'epoca del target nel dataset COMPLETO `all_labels`
                # La prima previsione che vogliamo fare è `training_epochs_count` + `gap_epochs` (i.e. 3h + 1min)
                
                # Costruiamo un set di test specifico per questo `current_gap_minutes`
                X_test_gap, y_true_gap = [], []
                
                # Iteriamo su tutti i dati, ma selezioniamo solo le sequenze che iniziano
                # in modo che il loro target cada *dopo* la finestra di training.
                # E il target deve essere esattamente a `current_gap_minutes` dopo la fine della sequenza di look_back.
                
                # Example:
                # Training Window: [0, ..., T-1]
                # Predict Target P_k (3h + k min)
                # Sequence for P_k should be [P_k - LOOK_BACK - gap_epochs, ..., P_k - gap_epochs - 1]
                # where gap_epochs corresponds to 'k' minutes.
                
                # Let's redefine. We predict `k` minutes *after* the end of the `LOOK_BACK` window.
                # The `LOOK_BACK` window can itself be at any point in the time series.
                
                # We want to predict for `prediction_features_full` starting from its beginning.
                # `prediction_scaled_features_full` starts at `training_epochs_count` in `all_features`.
                
                # So, a sequence `X_seq` for prediction is `prediction_scaled_features_full[i : i + LOOK_BACK]`.
                # Its target `y_target` is `prediction_labels_full[i + LOOK_BACK + gap_epochs]`.
                
                # We need to ensure that `i + LOOK_BACK + gap_epochs` is within bounds of `prediction_labels_full`.
                
                X_test_seq, y_test_true = experiment_logic.prepare_sequences(
                    prediction_scaled_features_full, prediction_labels_full, LOOK_BACK, gap_epochs
                )

                if len(X_test_seq) == 0:
                    # Se non ci sono sequenze per questo gap, incrementa il gap e riprova.
                    # print(f"  - Gap di {current_gap_minutes} min: Non abbastanza dati per creare sequenze. Salto.")
                    current_gap_minutes += PREDICTION_INCREMENT_MINUTES
                    continue

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
                    'num_predictions': len(y_test_true) # Numero di previsioni effettuate per questo gap
                })
                
                # Aggiorna le liste per la matrice di confusione aggregata
                all_y_true_for_cm.extend(y_test_true)
                all_y_pred_for_cm.extend(y_test_pred)
                
                current_gap_minutes += PREDICTION_INCREMENT_MINUTES
        
        # Genera una matrice di confusione aggregata per tutte le previsioni di questo soggetto
        if all_y_true_for_cm:
            title_cm_overall = (f"LSTM - Subject {subject_id}\n"
                                f"Train Window: {TRAINING_WINDOW_HOURS_PER_SUBJECT} hours\n"
                                f"Aggregated Predictions from {PREDICTION_INCREMENT_MINUTES} min onwards")
            
            filepath_cm_overall = os.path.join(fold_dir, f"cm_lstm_aggregated_subj_{subject_id}.pdf")
            utils.plot_confusion_matrix(all_y_true_for_cm, all_y_pred_for_cm, UNIQUE_CLASS_NAMES, title_cm_overall, filepath_cm_overall)
            print(f"Matrice di confusione aggregata per il soggetto {subject_id} salvata in: {filepath_cm_overall}")

    print(f"\n{'='*80}\n--- ESPERIMENTO COMPLETATO ---\n{'='*80}")
    if not final_results:
        print("Nessun risultato raccolto.")
        return

    results_df = pd.DataFrame(final_results)
    print("\n--- Risultati Complessivi ---")
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, 'subject_specific_lstm_results.csv'), index=False)

    print("\n--- Riepilogo Medie per Gap di Previsione ---")
    summary = results_df.groupby(['prediction_gap_minutes'])['accuracy'].agg(['mean', 'std', 'count'])
    print(summary)
    summary.to_csv(os.path.join(OUTPUTS_DIR, 'subject_specific_lstm_summary_by_gap.csv'))
    
    print(f"\nRisultati salvati nella directory '{OUTPUTS_DIR}'")

if __name__ == '__main__':
    # Puoi specificare un sottoinsieme di soggetti per test più rapidi
    # subjects_for_experiment = [0, 1] # Esempio: primi due soggetti
    subjects_for_experiment = list(range(3)) # Per testare con 3 soggetti
    # subjects_for_experiment = list(range(83)) # Tutti i soggetti (richiede molto tempo)
    
    run_experiment(all_subject_ids=subjects_for_experiment)