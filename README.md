# Predizione degli Stadi del Sonno con Validazione "Leave-One-Subject-Out"

Questo progetto implementa un sistema avanzato in Python per la predizione degli stadi del sonno, utilizzando un approccio di validazione incrociata di tipo **"Leave-One-Subject-Out" (LOSO)**. Questo metodo garantisce che il modello venga testato su dati di un soggetto mai visto durante l'addestramento, fornendo una stima robusta delle sue performance nel mondo reale.

Il sistema predice gli stadi futuri del sonno a diversi intervalli di tempo (es. 30, 60, 90, 120 minuti) e genera un'analisi dettagliata delle performance per ogni soggetto testato.

## Architettura e Modelli

Il sistema utilizza un approccio a doppio modello:

1.  **LSTM (Long Short-Term Memory) Network (PyTorch)**: Analizza sequenze di epoche passate per predire uno stadio del sonno futuro. Ideale per catturare le dipendenze temporali nei pattern di sonno.
2.  **Random Forest Classifier (Scikit-learn)**: Funge da modello di riferimento, classificando lo stadio del sonno *attuale* basandosi sulle feature di una singola epoca.

Questo lavoro è un'implementazione pratica e una continuazione dei concetti presentati nel seguente articolo di ricerca:
> Lozzi, D., Di Matteo, A., Mattei, E., Cipriani, A., Caianiello, P., Mignosi, F., & Placidi, G. (2024, October). ASIS: A Smart Alarm Clock Based on Deep Learning for the Safety of Night Workers. In *2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)* (pp. 1129-1134). IEEE.

## Funzionalità Principali

-   **Validazione Robusta**: Implementa una pipeline di cross-validazione "Leave-One-Subject-Out" (LOSO).
-   **Predizione Multi-Gap**: Addestra un singolo modello (con un gap di 30 min) e lo valuta su più orizzonti temporali (30, 60, 90, 120 min).
-   **Reportistica Automatica**: Per ogni fold della validazione (cioè per ogni soggetto usato come test), il sistema genera automaticamente:
    -   Un file **JSON** con la cronologia di addestramento (loss e accuracy per epoca).
    -   Un **grafico PDF** che mostra l'andamento di loss e accuracy durante l'addestramento.
    -   **Matrici di confusione in PDF** sia per il Random Forest sia per l'LSTM (per ogni gap di previsione), con titoli dettagliati.
-   **Risultati Aggregati**: Al termine dell'esperimento, vengono creati file CSV con le performance medie e per singolo fold.
-   **Download Automatico dei Dati**: Utilizza `mne` per scaricare e gestire il dataset PhysioNet Sleep-EDF.

## Struttura del Progetto

-   `config.py`: Parametri globali (gap di previsione, epoche, etc.).
-   `data_loader.py`: Caricamento e gestione dei dati del dataset.
-   `feature_extractor.py`: Calcolo delle feature Power Spectral Density (PSD).
-   `models.py`: Definizione dei modelli PyTorch (LSTM) e Scikit-learn (RF).
-   `train.py`: Funzioni per l'addestramento dei modelli su un set di soggetti.
-   `predict.py`: Funzioni per la valutazione dei modelli su un soggetto di test.
-   `run_loso_experiment.py`: **Script principale** per orchestrare l'intero esperimento LOSO.
-   `utils.py`: Funzioni di utilità per grafici, matrici di confusione e salvataggio file.
-   `requirements.txt`: Dipendenze Python del progetto.

## Installazione

1.  **Clona il repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Crea un ambiente virtuale (consigliato) e installa le dipendenze:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    **Nota:** Il dataset verrà scaricato automaticamente alla prima esecuzione.

## Come Eseguire l'Esperimento

L'intero processo è gestito dallo script `run_loso_experiment.py`.

1.  **Configura l'Esperimento**:
    Apri `run_loso_experiment.py` e modifica la lista `subjects_for_experiment`. Per un test rapido, puoi usare un piccolo numero di soggetti (es. `list(range(4))`). Per l'esperimento completo, usa `list(range(83))`.

2.  **Avvia l'Esperimento**:
    Esegui lo script dal terminale:
    ```bash
    python run_loso_experiment.py
    ```

## Analisi dei Risultati

Al termine dell'esecuzione, troverai una nuova directory `outputs/`. Al suo interno, ci sarà una sottodirectory per ogni fold dell'esperimento (es. `fold_test_subject_0/`), contenente:

-   `training_history.json`: Dati di loss/accuracy per epoca.
-   `training_history.pdf`: Grafico dell'andamento dell'addestramento.
-   `confusion_matrix_rf.pdf`: Matrice di confusione per il Random Forest.
-   `confusion_matrix_lstm_30min.pdf`, `..._60min.pdf`, etc.: Matrici di confusione per l'LSTM per ogni gap di valutazione.

Nella directory `outputs/` principale, troverai anche due file CSV riassuntivi:

-   `loso_results_per_fold.csv`: Metriche di accuratezza dettagliate per ogni soggetto di test.
-   `loso_results_summary.csv`: Medie e deviazioni standard di tutte le metriche, per una visione d'insieme delle performance del modello.
