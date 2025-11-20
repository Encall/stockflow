from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# Scaler configurations
SCALER_OPTIONS = {
    "minmax": MinMaxScaler(),
    "standard": StandardScaler(),
    "robust": RobustScaler(),
    "maxabs": MaxAbsScaler(),
    "none": None
}

# Dataset hyperparameters
DATASET_PARAMS = {
    "seq_len": [30, 50, 60, 90, 120],
    "scaler": list(SCALER_OPTIONS.keys())
}

# Model hyperparameters
LSTM_PARAMS = {
    "hidden_size": [32, 64, 128, 256],
    "num_layers": [1, 2, 3, 4],
    "output_size": [1],
    "dropout": [0.1, 0.2, 0.3, 0.4]
}

GRU_PARAMS = {
    "hidden_size": [32, 64, 128, 256],
    "num_layers": [1, 2, 3, 4],
    "output_size": [1],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "bidirectional": [False, True]
}

NBERT_PARAMS = {
    "output_size": [1],
    "hidden_dim": [64, 128, 256],
    "n_blocks": [2, 3, 4, 5],
    "n_layers": [2, 3, 4, 5],
    "dropout": [0.1, 0.2, 0.3]
}

TRANSFORMER_PARAMS = {
    "d_model": [32, 64, 128],
    "nhead": [2, 4, 8],
    "num_layers": [1, 2, 3, 4],
    "dim_feedforward": [64, 128, 256, 512],
    "dropout": [0.1, 0.2, 0.3],
    "output_size": [1]
}

# Training hyperparameters
TRAINING_PARAMS = {
    "lr": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "epochs": [30, 50, 100],
    "batch_size": [16, 32, 64, 128],
    "loss_fn": ["MSE", "MAE", "Huber"]
}

MODEL_PARAMS = {
    "LSTM": LSTM_PARAMS,
    "GRU": GRU_PARAMS,
    "NBERT": NBERT_PARAMS,
    "Transformer": TRANSFORMER_PARAMS
}

# Default configurations for staged tuning
DEFAULT_DATASET_CONFIG = {
    "seq_len": 60,
    "scaler": "minmax"
}

DEFAULT_TRAINING_CONFIG = {
    "lr": 0.001,
    "epochs": 30,
    "batch_size": 32,
    "loss_fn": "MSE"
}

# Get default model configs (first value of each parameter)
def get_default_model_config(model_type: str) -> dict:
    """Get default configuration for a model type"""
    params = MODEL_PARAMS[model_type]
    return {key: values[0] for key, values in params.items()}