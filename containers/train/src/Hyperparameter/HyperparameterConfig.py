from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

SCALER_OPTIONS = {
    "standard": StandardScaler()
}

DATASET_PARAMS = {
    "seq_len": [30],
    "scaler": list(SCALER_OPTIONS.keys())
}

LSTM_PARAMS = {
    "hidden_size": [32, 64, 128],
    "num_layers": [3, 4],
    "output_size": [1],
    "dropout": [0.1, 0.2, 0.3]
}

GRU_PARAMS = {
    "hidden_size": [32, 64, 128],
    "num_layers": [3, 4],
    "output_size": [1],
    "dropout": [0.1, 0.2, 0.3],
    "bidirectional": [False, True]
}

NBERT_PARAMS = {
    "output_size": [1],
    "hidden_dim": [32, 64, 128],
    "n_blocks": [2, 3, 4],
    "n_layers": [5],
    "dropout": [0.1, 0.2, 0.3]
}

TRANSFORMER_PARAMS = {
    "d_model": [32, 64, 128],
    "nhead": [2, 4, 8],
    "num_layers": [2, 3, 4],
    "dim_feedforward": [64, 128],
    "dropout": [0.1, 0.2, 0.3],
    "output_size": [1]
}

TRAINING_PARAMS = {
    "lr": [0.005, 0.01],
    "epochs": [30],
    "batch_size": [64],
    "loss_fn": ["MSE", "MAE"],
    "patience": [5]
}

MODEL_PARAMS = {
    "LSTM": LSTM_PARAMS,
    "GRU": GRU_PARAMS,
    "NBERT": NBERT_PARAMS,
    "Transformer": TRANSFORMER_PARAMS
}

DEFAULT_TRAINING_CONFIG = {
    "lr": 0.001,
    "epochs": 5,
    "batch_size": 128,
    "loss_fn": "MSE",
    "patience": 5
}

def get_default_model_config(model_type: str) -> dict:
    params = MODEL_PARAMS[model_type]
    return {key: values[0] for key, values in params.items()}