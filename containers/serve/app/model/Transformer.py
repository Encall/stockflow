import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)          # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even
        pe[:, 1::2] = torch.cos(position * div_term)   # odd

        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for batch_first
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_size: int = 1,
        pkl_path: str | None = None,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, output_size)

        if pkl_path:
            self.load_model(pkl_path)
            self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)          # (batch, seq_len, d_model)

        x = self.pos_encoder(x)         # (batch, seq_len, d_model)

        enc_out = self.encoder(x)       # (batch, seq_len, d_model)

        last_hidden = enc_out[:, -1, :]  # (batch, d_model)
        price_pred = self.fc_out(last_hidden)  # (batch, output_size)
        return price_pred

    def load_model(self, path: str):
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)