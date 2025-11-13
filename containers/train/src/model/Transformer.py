import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_heads=4, num_layers=2,
                 output_size=1, dropout=0.1, pkl_path=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)

        self.positional_encoding = self._generate_positional_encoding(500, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, output_size)

        if pkl_path:
            self.load_model(pkl_path)
            self.eval()

    def _generate_positional_encoding(self, max_len, hidden_size):
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, hidden_size, dtype=torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / hidden_size)
        angle_rads = pos * angle_rates
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        pe = pe.unsqueeze(0)
        return pe  # shape (1, max_len, hidden_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)
        out = self.transformer(x)
        last_hidden = out[:, -1, :]
        price_pred = self.fc(last_hidden)
        return price_pred

    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
