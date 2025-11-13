import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

import pandas as pd


class MultiFeaturePriceDataset(Dataset):
    def __init__(self, data, feature_cols, target_col="Close", seq_len=60, scaler=None):
        self.seq_len = seq_len
        self.X_all = data[feature_cols].values
        self.y_all = data[target_col].values.reshape(-1, 1)

        if scaler is not None:
            self.x_scaler = scaler
            self.y_scaler = scaler
            self.X_all = self.x_scaler.fit_transform(self.X_all)
            self.y_all = self.y_scaler.fit_transform(self.y_all)

        X_list, y_list = [], []

        n = self.X_all.shape[0]
        for i in range(n - seq_len):
            X_seq = self.X_all[i : i+seq_len]
            y_val = self.y_all[i+seq_len]

            X_list.append(X_seq)
            y_list.append(y_val)

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
