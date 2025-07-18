import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

def create_lstm_model(input_size, hidden_size, num_layers, num_classes, dropout):
    """
    Crea un modello LSTM per la previsione degli stadi del sonno utilizzando PyTorch.
    """
    class LSTMClassifier(nn.Module):
        def __init__(self):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            out = self.dropout(last_output)
            out = self.fc(out)
            return out

    model = LSTMClassifier()
    return model


def get_classical_classifier(n_estimators=100, random_state=42):
    """
    Restituisce un classificatore classico pre-configurato.
    """
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1 
    )
    return classifier