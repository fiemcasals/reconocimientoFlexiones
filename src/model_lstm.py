# src/model_lstm.py
import torch.nn as nn

class PoseLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=256, num_layers=1, num_classes=2):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # último paso
        return out
