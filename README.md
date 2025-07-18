# Predizione Avanzata degli Stadi del Sonno con Addestramento Incrementale e Validazione LOSO

Questo progetto implementa una pipeline di valutazione rigorosa per modelli di predizione degli stadi del sonno, basata su un approccio di **addestramento incrementale** e validazione incrociata **"Leave-One-Subject-Out" (LOSO)**.

Lo scopo è simulare uno scenario realistico in cui un modello viene addestrato con una quantità crescente di dati (es. i primi 30, 60, 90 minuti di sonno) per valutare come le sue performance evolvono nel tempo.

## Riferimento Scientifico

Questo lavoro è un'implementazione pratica e una continuazione dei concetti presentati nel seguente articolo di ricerca:
> Lozzi, D., Di Matteo, A., Mattei, E., Cipriani, A., Caianiello, P., Mignosi, F., & Placidi, G. (2024, October). ASIS: A Smart Alarm Clock Based on Deep Learning for the Safety of Night Workers. In *2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)* (pp. 1129-1134). IEEE.

### Citazione BibTeX
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

## Logica Sperimentale

L'esperimento segue una struttura di validazione LOSO. Per ogni soggetto del dataset (usato a rotazione come "soggetto di test"):

1.  **Finestre di Addestramento Crescenti**: Vengono definite diverse finestre temporali (es. 30, 60, 90, 120 minuti).
2.  **Addestramento per Finestra**: Per ogni finestra temporale, vengono addestrati due modelli distinti utilizzando i dati di tutti gli altri soggetti ("soggetti di training"):
    * **Random Forest (Classificazione Istantanea)**:
        * **Addestramento**: Utilizza i dati dei soggetti di training *fino alla fine della finestra corrente* (es. i primi 30 minuti).
        * **Valutazione**: Classifica gli stadi del sonno del soggetto di test *all'interno della stessa finestra* (es. i suoi primi 30 minuti).
    * **LSTM (Predizione Futura)**:
        * **Addestramento**: Utilizza i dati dei soggetti di training *fino alla fine della finestra corrente*.
        * **Valutazione**: Predice gli stadi del sonno futuri per l'**intera notte** del soggetto di test.

Questo approccio permette di rispondere a domande come: "Quanti dati servono per ottenere una predizione affidabile?" e "Come si comporta il modello su dati mai visti prima?".

## Funzionalità Principali

-   **Validazione LOSO**: Garantisce una stima robusta e generalizzabile delle performance.
-   **Addestramento Incrementale**: I modelli vengono ri-addestrati su finestre di dati crescenti.
-   **Valutazione a Doppio Task**: Analisi sia della classificazione istantanea (RF) sia della predizione futura (LSTM).
-   **Reportistica Automatica e Dettagliata**: Per ogni configurazione (soggetto di test, finestra di training), il sistema genera:
    -   Grafici **PDF** con le curve di apprendimento (loss e accuracy).
    -   Matrici di confusione **PDF** per ogni modello, con titoli che specificano le condizioni dell'esperimento.
-   **Risultati Aggregati**: Al termine, vengono creati file **CSV** con i risultati dettagliati e un riepilogo delle performance medie.

## Struttura del Progetto

-   `config.py`: Parametri globali, incluse le finestre di addestramento.
-   `data_loader.py`: Caricamento dati, con opzione per limitare le epoche.
-   `feature_extractor.py`: Estrazione delle feature PSD.
-   `models.py`: Definizione dei modelli.
-   `experiment_logic.py`: **Nuovo file** che incapsula la logica di addestramento e valutazione per una singola configurazione.
-   `run_advanced_loso.py`: **Script principale** che orchestra l'intero esperimento LOSO avanzato.
-   `utils.py`: Funzioni di utilità per la creazione di grafici e il salvataggio dei risultati.
-   `requirements.txt`: Dipendenze del progetto.

## Installazione

1.  **Clona il repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Crea un ambiente virtuale e installa le dipendenze:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Come Eseguire l'Esperimento

L'intero processo è gestito dallo script `run_advanced_loso.py`.

1.  **Configura l'Esperimento**:
    Apri `run_advanced_loso.py` e modifica la lista `subjects_for_experiment`.

    **ATTENZIONE**: Questo esperimento è **estremamente intensivo** dal punto di vista computazionale. Si consiglia vivamente di iniziare con un piccolo numero di soggetti (es. `list(range(3))`) per verificare che la pipeline funzioni correttamente.

2.  **Avvia l'Esperimento**:
    Esegui lo script dal terminale:
    ```bash
    python run_advanced_loso.py
    ```

## Analisi dei Risultati

Al termine, verrà creata una directory `outputs_advanced/`. Al suo interno:

-   **Sottodirectory per Fold**: Una cartella per ogni soggetto di test (es. `fold_test_subject_0/`), contenente tutti i grafici (matrici di confusione e cronologie di addestramento) generati per quel soggetto nelle varie configurazioni.
-   **File CSV Riepilogativi**:
    -   `advanced_loso_results.csv`: Contiene i risultati grezzi (accuratezza) per ogni singola esecuzione (soggetto, modello, finestra di training).
    -   `advanced_loso_summary.csv`: Fornisce una vista aggregata, mostrando media e deviazione standard delle performance per ogni combinazione di modello e finestra di training. Questo è il file più importante per trarre conclusioni generali.
