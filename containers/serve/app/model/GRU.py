import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pkl_path: str | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_dropout = nn.Dropout(dropout)

        fc_in_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in_dim, output_size)

        if pkl_path:
            self.load_model(pkl_path)
            self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)

        if self.bidirectional:
            last_forward  = h_n[-2, :, :]
            last_backward = h_n[-1, :, :]
            last_hidden = torch.cat([last_forward, last_backward], dim=-1)
        else:
            last_hidden = h_n[-1, :, :] 

        last_hidden = self.output_dropout(last_hidden)

        price_pred = self.fc(last_hidden)
        return price_pred

    def load_model(self, path: str):
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
