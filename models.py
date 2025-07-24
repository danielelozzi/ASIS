# models.py

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

# Nuova classe del modello LSTM con MLP per feature temporale
class LSTMTemporalPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout, mlp_hidden_size, mlp_dropout):
        super(LSTMTemporalPredictor, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout if num_layers > 1 else 0)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        
        # MLP layers
        # L'input del MLP sarà l'output dell'LSTM (hidden_size) + 1 (per il time_feature normalizzato)
        self.mlp_fc1 = nn.Linear(hidden_size + 1, mlp_hidden_size)
        self.mlp_relu = nn.ReLU()
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        
        # Output layer
        self.fc_out = nn.Linear(mlp_hidden_size, num_classes)

    def forward(self, x_lstm, time_feature):
        # x_lstm: (batch_size, seq_len, input_size)
        # time_feature: (batch_size, 1)

        # Passa attraverso l'LSTM
        lstm_out, _ = self.lstm(x_lstm)
        # Prendi l'ultimo output della sequenza
        last_lstm_output = lstm_out[:, -1, :] # (batch_size, hidden_size)
        last_lstm_output = self.lstm_dropout(last_lstm_output) # Applica dropout all'output LSTM

        # Concatena l'output dell'LSTM con la feature temporale
        combined_features = torch.cat((last_lstm_output, time_feature), dim=1) # (batch_size, hidden_size + 1)
        
        # Passa attraverso il MLP
        mlp_out = self.mlp_fc1(combined_features)
        mlp_out = self.mlp_relu(mlp_out)
        mlp_out = self.mlp_dropout(mlp_out) # Applica dropout al MLP

        # Output finale
        out = self.fc_out(mlp_out)
        return out

def create_lstm_temporal_predictor(input_size, hidden_size, num_layers, num_classes, lstm_dropout, mlp_hidden_size, mlp_dropout):
    return LSTMTemporalPredictor(input_size, hidden_size, num_layers, num_classes, lstm_dropout, mlp_hidden_size, mlp_dropout)


def get_classical_classifier(n_estimators=100, random_state=42, class_weight='balanced'):
    """
    Restituisce un classificatore Random Forest.
    'class_weight' è impostato su 'balanced' per gestire lo squilibrio delle classi.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )