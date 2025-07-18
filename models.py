# models.py

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

def create_lstm_model(input_size, hidden_size, num_layers, num_classes, dropout):
    class LSTMClassifier(nn.Module):
        def __init__(self):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            out = self.dropout(last_output)
            out = self.fc(out)
            return out
    return LSTMClassifier()

def get_classical_classifier(n_estimators=100, random_state=42, class_weight='balanced'):
    """
    Restituisce un classificatore Random Forest.
    'class_weight' è impostato su 'balanced' per gestire lo squilibrio delle classi.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight, # <-- MODIFICA CHIAVE
        n_jobs=-1
    )