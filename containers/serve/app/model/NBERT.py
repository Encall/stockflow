# model/NBEATS.py
import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,    
        forecast_len: int, 
        hidden_dim: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.fc_stack = nn.Sequential(*layers)

        self.backcast_linear = nn.Linear(hidden_dim, input_dim)
        self.forecast_linear = nn.Linear(hidden_dim, forecast_len)

    def forward(self, x: torch.Tensor):
        h = self.fc_stack(x)
        backcast = self.backcast_linear(h)
        forecast = self.forecast_linear(h)
        return backcast, forecast


class NBERT(nn.Module):
    def __init__(
        self,
        input_size: int,    
        seq_len: int,      
        output_size: int = 1,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        n_layers: int = 4,
        dropout: float = 0.1,
        pkl_path: str | None = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.backcast_dim = seq_len * input_size
        self.forecast_len = output_size

        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_dim=self.backcast_dim,
                forecast_len=self.forecast_len,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            )
            for _ in range(n_blocks)
        ])

        self.dropout = nn.Dropout(dropout)

        if pkl_path:
            self.load_model(pkl_path)
            self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        backcast = x.reshape(batch_size, -1)

        forecast = torch.zeros(
            batch_size, self.forecast_len, device=x.device, dtype=x.dtype
        )
        for block in self.blocks:
            b, f = block(backcast)
            backcast = backcast - b
            forecast = forecast + f

        return forecast

    def load_model(self, path: str):
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
