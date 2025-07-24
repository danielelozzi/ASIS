Predizione Avanzata degli Stadi del Sonno con Addestramento Specifico per Soggetto e Previsione Temporale

Questo progetto implementa una pipeline di valutazione rigorosa per modelli di predizione degli stadi del sonno, focalizzandosi su un approccio di **addestramento specifico per soggetto** e previsione con **gap temporali crescenti**.

Lo scopo è simulare uno scenario realistico di monitoraggio continuo del sonno, dove un modello LSTM leggero viene addestrato sulle prime ore di sonno di un individuo e poi utilizzato per prevedere gli stadi successivi a intervalli specifici fino al risveglio.
#
 #  Riferimento Scientifico
#
  Questo lavoro è un'implementazione pratica e una continuazione dei concetti presentati nel seguente articolo di ricerca, adattato per esplorare la predizione temporale per singolo soggetto:
  > Lozzi, D., Di Matteo, A., Mattei, E., Cipriani, A., Caianiello, P., Mignosi, F., & Placidi, G. (2024, October). ASIS: A Smart Alarm Clock Based on Deep Learning for the Safety of Night Workers. In *2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)* (pp. 1129--1134). IEEE.
#
 ##  Citazione BibTeX
 ```bibtex
 @inproceedings{lozzi2024asis,
   title={ASIS: A Smart Alarm Clock Based on Deep Learning for the Safety of Night Workers},
   author={Lozzi, Daniele and Di Matteo, Alessandro and Mattei, Enrico and Cipriani, Alessia and Caianiello, Pasquale and Mignosi, Filippo and Placidi, Giuseppe},
   booktitle={2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)},
   pages={1129--1134},
   year={2024},
   organization={IEEE}
 }
 ```
#
 #  Logica Sperimentale
#
  L'esperimento si concentra sull'analisi delle prestazioni di un modello LSTM addestrato e valutato *per ogni singolo soggetto*. Per ciascun soggetto del dataset:
#
 1.   **Suddivisione dei Dati**: I dati del sonno del soggetto vengono divisi in due parti:
     *  **Finestra di Addestramento Iniziale**: Le prime **3 ore** di sonno vengono utilizzate per addestrare un modello LSTM.
     *  **Dati di Previsione**: Il resto della notte (dopo le prime 3 ore) viene utilizzato per valutare le capacità predittive del modello.
#
 2.   **Addestramento LSTM Leggero**: Viene addestrato un modello **LSTM leggero** (con un numero ridotto di unità e dropout) esclusivamente sui dati della finestra di addestramento iniziale di quel soggetto. Questo simula uno scenario in cui il modello è adattato specificamente all'individuo con una quantità limitata di dati iniziali.
#
 3.   **Valutazione con Gap Temporali Crescenti**: Il modello LSTM addestrato viene poi utilizzato per effettuare previsioni sui dati rimanenti del soggetto. Le previsioni vengono effettuate per stadi del sonno che si trovano a **3 ore e 1 minuto, 3 ore e 2 minuti**, e così via, fino al risveglio del soggetto. Questo permette di valutare come l'accuratezza della predizione decresce all'aumentare del "gap" temporale tra i dati di osservazione e il momento della predizione.
#
  Questo approccio permette di rispondere a domande come: "Quanto bene un modello leggero può adattarsi a un singolo individuo con poche ore di dati?" e "Quanto a lungo nel futuro è possibile prevedere gli stadi del sonno con un'accuratezza accettabile?".
#
 #  Funzionalità Principali
#
 -    **Addestramento Soggetto-Specifico**: Un modello LSTM viene addestrato individualmente per ogni soggetto, utilizzando solo le prime 3 ore dei suoi dati.
 -    **Valutazione a Previsione Temporale**: Analisi della capacità dell'LSTM di prevedere stadi del sonno futuri con incrementi di 1 minuto dopo la finestra di training.
 -    **Reportistica Automatica e Dettagliata**: Per ogni soggetto, il sistema genera:
     -    Grafici **PDF** con le curve di apprendimento (loss e accuracy) del training LSTM.
     -    Matrici di confusione **PDF** aggregate per tutte le previsioni effettuate su quel soggetto.
 -    **Risultati Aggregati**: Al termine, vengono creati file **CSV** con i risultati dettagliati (accuratezza per ogni gap di previsione) e un riepilogo delle performance medie per ciascun gap.
#
 #  Struttura del Progetto
#
 -    `config.py`: Parametri globali, inclusi i nuovi parametri per la finestra di training fissa e l'incremento di previsione.
 -    `data_loader.py`: Caricamento dati del sonno.
 -    `feature_extractor.py`: Estrazione delle feature PSD.
 -    `models.py`: Definizione dei modelli (LSTM leggero e Random Forest, anche se quest'ultimo non è più il focus principale).
 -    `experiment_logic.py`: Contiene funzioni di utilità come `prepare_sequences`, ma la logica di training/valutazione è ora in `run_advanced_loso.py`.
 -    `run_advanced_loso.py`: **Script principale** che orchestra l'intero esperimento di training soggetto-specifico e previsione temporale.
 -    `utils.py`: Funzioni di utilità per la creazione di grafici e il salvataggio dei risultati.
 -    `requirements.txt`: Dipendenze del progetto.
#
 #  Installazione
#
 1.   **Clona il repository:**
     ```bash
     git clone <your-repository-url>
     cd <your-repository-name>
     ```
#
 2.   **Crea un ambiente virtuale e installa le dipendenze:**
     ```bash
     python -m venv venv
     source venv/bin/activate   Su Windows: venv\Scripts\activate
     pip install -r requirements.txt
     ```
#
 #  Come Eseguire l'Esperimento
#
  L'intero processo è gestito dallo script `run_advanced_loso.py`.
#
 1.   **Configura l'Esperimento**:
      Apri `run_advanced_loso.py` e modifica la lista `subjects_for_experiment`.
#
      **ATTENZIONE**: Questo esperimento, pur con modelli più leggeri, può comunque essere **intensivo** dal punto di vista computazionale a seconda del numero di soggetti. Si consiglia vivamente di iniziare con un piccolo numero di soggetti (es. `list(range(3))`) per verificare che la pipeline funzioni correttamente.
#
 2.   **Avvia l'Esperimento**:
      Esegui lo script dal terminale:
     ```bash
     python run_advanced_loso.py
     ```
#
 #  Analisi dei Risultati
#
  Al termine, verrà creata una directory `outputs_subject_specific/`. Al suo interno:
#
 -    **Sottodirectory per Soggetto**: Una cartella per ogni soggetto valutato (es. `subject_0/`), contenente i grafici della cronologia di addestramento LSTM e le matrici di confusione aggregate per quel soggetto.
 -    **File CSV Riepilogativi**:
     -    `subject_specific_lstm_results.csv`: Contiene i risultati grezzi (accuratezza) per ogni singola previsione (soggetto, gap temporale).
     -    `subject_specific_lstm_summary_by_gap.csv`: Fornisce una vista aggregata, mostrando media, deviazione standard e numero di previsioni per ogni specifico "gap" temporale. Questo è il file più importante per trarre conclusioni generali sulla capacità predittiva nel tempo.