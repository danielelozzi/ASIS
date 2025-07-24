  Predizione Avanzata degli Stadi del Sonno con Addestramento Generale e Predizione Temporale

  Questo progetto implementa una pipeline di valutazione rigorosa per modelli di predizione degli stadi del sonno, focalizzandosi su un approccio di **addestramento generale su più soggetti** e previsione con **gap temporali crescenti** utilizzando un riferimento esplicito al tempo.

  Lo scopo è simulare uno scenario realistico di monitoraggio continuo del sonno, dove un modello LSTM leggero viene addestrato sulle prime ore di sonno di un pool di individui e poi utilizzato per prevedere gli stadi successivi a intervalli specifici fino al risveglio, tenendo conto esplicitamente del tempo trascorso.

#  Riferimento Scientifico

  Questo lavoro è un'implementazione pratica e una continuazione dei concetti presentati nel seguente articolo di ricerca, adattato per esplorare la predizione temporale con feature esplicite:
  > Lozzi, D., Di Matteo, A., Mattei, E., Cipriani, A., Caianiello, P., Mignosi, F., & Placidi, G. (2024, October). ASIS: A Smart Alarm Clock Based on Deep Learning for the Safety of Night Workers. In *2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)* (pp. 1129--1134). IEEE.

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

#  Logica Sperimentale

  L'esperimento si concentra sull'analisi delle prestazioni di un **modello LSTM generale** addestrato su un ampio pool di soggetti e valutato sulla sua capacità di generalizzazione e previsione temporale.

  La pipeline segue questi passi:

1.   **Suddivisione dei Soggetti (80% Training, 20% Test)**:
      L'intero set di soggetti viene diviso casualmente: l'80% viene destinato all'addestramento del modello, e il 20% viene riservato per il test finale (soggetti mai visti dal modello).

2.   **Addestramento del Modello LSTM Generale**:
      Un **modello LSTM leggero**, potenziato con un **Multilayer Perceptron (MLP)** per incorporare un riferimento temporale esplicito, viene addestrato.
      * **Dati di Addestramento**: Per ogni soggetto nel set di training (80%), vengono utilizzate le **prime 3 ore** di sonno. Questi dati aggregati formano il set di training per il modello generale.
      * **Previsione Incorporata**: Durante l'addestramento, l'LSTM impara a prevedere gli stadi del sonno un piccolo intervallo di tempo (es. 1 minuto) dopo l'ultima epoca della sua sequenza di input. L'MLP nel modello riceve anche una feature che codifica questo intervallo di tempo.

3.   **Valutazione con Gap Temporali Estesi (sul 20% dei soggetti di test)**:
      Il modello LSTM addestrato (generale) viene poi utilizzato per effettuare previsioni sui soggetti del set di test (quelli non visti durante l'addestramento).
      * Per ogni soggetto di test, il modello predice gli stadi del sonno a intervalli temporali specifici (es. **3 ore e 1 minuto, 3 ore e 2 minuti, 3 ore e 5 minuti, ... fino a 12 ore**) dall'inizio della loro registrazione del sonno.
      * Il **riferimento temporale** (la distanza tra il punto di previsione e l'inizio del sonno) viene passato esplicitamente all'MLP del modello, permettendo al modello di adattare la sua previsione in base al "quanto lontano nel futuro" sta cercando di predire.

  Questo approccio permette di rispondere a domande chiave come: "Quanto bene un modello addestrato su un pool di individui generalizza a soggetti nuovi?", e "Quanto a lungo nel futuro è possibile prevedere gli stadi del sonno mantenendo un'accuratezza accettabile, sfruttando un riferimento temporale esplicito?".

#  Funzionalità Principali

 -    **Addestramento Generale del Modello**: Un singolo modello LSTM viene addestrato su un ampio set di soggetti per massimizzare la generalizzazione.
 -    **Integrazione della Feature Temporale**: Il modello include un MLP che combina l'output dell'LSTM con un riferimento temporale esplicito, consentendo previsioni a diverse distanze nel futuro.
 -    **Valutazione a Previsione Temporale Estesa**: Analisi della capacità del modello di prevedere stadi del sonno futuri a intervalli di tempo fino a 12 ore.
 -    **Split Soggetti Riproducibile**: Utilizzo di un seed casuale per garantire la riproducibilità della divisione tra soggetti di training e test.
 -    **Bilanciamento delle Classi**: Il training utilizza un campionamento pesato per affrontare lo squilibrio delle classi negli stadi del sonno, migliorando le prestazioni sulle classi minoritarie.
 -    **Reportistica Automatica e Dettagliata**: Il sistema genera:
      -   Grafici **PDF** con le curve di apprendimento (loss e accuracy) del training del modello generale.
      -   Matrici di confusione **PDF** aggregate per tutte le previsioni effettuate su ogni soggetto di test.
 -    **Risultati Aggregati**: Al termine, vengono creati file **CSV** con i risultati dettagliati (accuratezza per ogni soggetto e gap temporale) e un riepilogo delle performance medie per ciascun gap di previsione sull'intero set di test.

#  Struttura del Progetto

 -    `config.py`: Parametri globali, inclusi i nuovi parametri per la split dei soggetti, i gap di previsione estesi e i parametri del MLP.
 -    `data_loader.py`: Caricamento dati del sonno.
 -    `feature_extractor.py`: Estrazione delle feature PSD.
 -    `models.py`: Definizione della nuova architettura del modello `LSTMTemporalPredictor` che include LSTM e MLP.
 -    `experiment_logic.py`: Contiene funzioni di utilità come `prepare_sequences`, modificata per includere la feature temporale.
 -    `run_advanced_loso.py`: **Script principale** che orchestra l'intero esperimento: split dei soggetti, caricamento e pre-elaborazione dei dati, addestramento del modello generale, e valutazione sui soggetti di test con analisi dei risultati.
 -    `utils.py`: Funzioni di utilità per la creazione di grafici e il salvataggio dei risultati.
 -    `requirements.txt`: Dipendenze del progetto.

#  Installazione

1.   **Clona il repository:**
     ```bash
     git clone <your-repository-url>
     cd <your-repository-name>
     ```

2.   **Crea un ambiente virtuale e installa le dipendenze:**
     ```bash
     python -m venv venv
     source venv/bin/activate   Su Windows: venv\Scripts\activate
     pip install -r requirements.txt
     ```

#  Come Eseguire l'Esperimento

  L'intero processo è gestito dallo script `run_advanced_loso.py`.

1.   **Configura l'Esperimento**:
      Apri `run_advanced_loso.py` e modifica la lista `all_subjects` alla fine dello script.

      **ATTENZIONE**: Questo esperimento è **molto intensivo** dal punto di vista computazionale, specialmente con tutti i soggetti e i numerosi punti di previsione. Si consiglia vivamente di iniziare con un piccolo numero di soggetti (es. `list(range(10))` o anche meno) per verificare che la pipeline funzioni correttamente prima di eseguire l'esperimento completo.

2.   **Avvia l'Esperimento**:
      Esegui lo script dal terminale:
     ```bash
     python run_advanced_loso.py
     ```

#  Analisi dei Risultati

  Al termine, verrà creata una directory `outputs_general_prediction_with_time_feature/`. Al suo interno:

 -    **Training del Modello Generale**:
      -   `general_model/lstm_general_model.pth`: Il modello LSTM addestrato.
      -   `general_model/scaler_general_model.pkl`: Lo scaler utilizzato per normalizzare le feature.
      -   `lstm_general_training_history.pdf`: Grafico della cronologia di training del modello generale.

 -    **Sottodirectory per Soggetto di Test**: Una cartella per ogni soggetto di test (es. `test_subject_0/`), contenente la matrice di confusione aggregata per tutte le previsioni effettuate su quel soggetto.

 -    **File CSV Riepilogativi (Test Set)**:
      -   `general_model_test_results_detailed.csv`: Contiene i risultati dettagliati (accuratezza, etichetta vera, etichetta predetta) per ogni singola previsione per ogni soggetto di test.
      -   `general_model_test_summary_by_gap.csv`: Fornisce una vista aggregata, mostrando media, deviazione standard e numero di previsioni (e soggetti contributori) per ogni specifico "target temporale" di previsione. Questo è il file più importante per trarre conclusioni generali sulla capacità predittiva del modello nel tempo.