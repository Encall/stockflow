import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pandas as pd


class SingleAssetDataset(Dataset):
    def __init__(self, data, feature_cols, target_col="Close", seq_len=60, scaler=None):
        self.seq_len = seq_len
        # Handle both string and list for target_col
        target = target_col[0] if isinstance(target_col, list) else target_col
        
        self.X_all = data[feature_cols].values
        self.y_all = data[target].values.reshape(-1, 1)

        if scaler is not None:
            # Use separate scalers for features and target
            self.x_scaler = scaler
            self.y_scaler = type(scaler)()  # Create new instance of same scaler type
            self.X_all = self.x_scaler.fit_transform(self.X_all)
            self.y_all = self.y_scaler.fit_transform(self.y_all)

        X_list, y_list = [], []

        n = self.X_all.shape[0]
        for i in range(n - seq_len):
            X_seq = self.X_all[i : i + seq_len]
            y_val = self.y_all[i + seq_len]

            X_list.append(X_seq)
            y_list.append(y_val)

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiAssetDataset(Dataset):
    def __init__(self, data, feature_cols, target_col="Close", seq_len=60, scaler=None):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        # Handle both string and list for target_col
        self.target_col = target_col[0] if isinstance(target_col, list) else target_col

        X_windows = []
        y_windows = []

        per_ticker_X = {}
        per_ticker_y = {}

        for ticker, df in data.items():
            df = df.copy()

            X_all = df[feature_cols].values
            y_all = df[self.target_col].values.reshape(-1, 1)

            per_ticker_X[ticker] = X_all
            per_ticker_y[ticker] = y_all

        if scaler is not None:
            # Use separate scalers for features and target
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            self.x_scaler = scaler
            # Create a new instance of the same scaler type for y
            self.y_scaler = type(scaler)()

            all_X = np.vstack(list(per_ticker_X.values()))
            all_y = np.vstack(list(per_ticker_y.values()))

            self.x_scaler.fit(all_X)
            self.y_scaler.fit(all_y)
        else:
            self.x_scaler = None
            self.y_scaler = None

        for ticker in data.keys():
            X_all = per_ticker_X[ticker]
            y_all = per_ticker_y[ticker]

            if self.x_scaler is not None:
                X_all = self.x_scaler.transform(X_all)
                y_all = self.y_scaler.transform(y_all)

            n = X_all.shape[0]
            if n <= seq_len:
                continue

            for i in range(n - seq_len):
                X_seq = X_all[i : i + seq_len]
                y_val = y_all[i + seq_len]

                X_windows.append(X_seq)
                y_windows.append(y_val)

        self.X = torch.tensor(np.array(X_windows), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_windows), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
