import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, pkl_path=None):
        
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

        if pkl_path:
            self.load_model(pkl_path)
            self.eval()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        price_pred = self.fc(last_hidden)
        return price_pred
    
    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
