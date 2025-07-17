# models.py

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from config import LSTM_UNITS, LSTM_DROPOUT, LABEL_MAP

def create_lstm_model(input_size, hidden_size, num_layers, num_classes, dropout):
    """
    Crea un modello LSTM per la previsione degli stadi del sonno utilizzando PyTorch.

    Args:
        input_size (int): Il numero di feature in input (n_features).
        hidden_size (int): Il numero di unità nello strato LSTM.
        num_layers (int): Il numero di strati LSTM.
        num_classes (int): Il numero di stadi del sonno da classificare.
        dropout (float): Il tasso di dropout da applicare.

    Returns:
        torch.nn.Module: Il modello LSTM.
    """
    class LSTMClassifier(nn.Module):
        def __init__(self):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True, # Aspetta input con forma (batch, seq, feature)
                dropout=dropout if num_layers > 1 else 0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            # L'output di LSTM è (output, (h_n, c_n))
            # Usiamo solo l'output dell'ultimo time step
            lstm_out, _ = self.lstm(x)
            
            # Prendi l'output dell'ultimo time step
            last_output = lstm_out[:, -1, :]
            
            out = self.dropout(last_output)
            out = self.fc(out)
            return out

    model = LSTMClassifier()
    return model


def get_classical_classifier(n_estimators=100, random_state=42):
    """
    Restituisce un classificatore classico pre-configurato.

    Args:
        n_estimators (int): Il numero di alberi nella foresta.
        random_state (int): Seme per la riproducibilità.

    Returns:
        sklearn.ensemble.RandomForestClassifier: Un classificatore Random Forest.
    """
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1 # Usa tutti i core disponibili
    )
    return classifier

if __name__ == '__main__':
    # Esempio di utilizzo e visualizzazione dei modelli
    
    # --- Esempio LSTM con PyTorch ---
    print("--- Architettura Modello LSTM (PyTorch) ---")
    # Parametri di esempio
    look_back = 10
    n_features = 10
    num_sleep_stages = len(LABEL_MAP)
    
    # Crea il modello LSTM
    lstm_model = create_lstm_model(
        input_size=n_features,
        hidden_size=LSTM_UNITS,
        num_layers=1,
        num_classes=num_sleep_stages,
        dropout=LSTM_DROPOUT
    )
    
    # Stampa l'architettura del modello
    print(lstm_model)
    
    # --- Esempio Classificatore Classico ---
    print("\n\n--- Classificatore Classico ---")
    classical_model = get_classical_classifier()
    print("Modello scelto:", classical_model)