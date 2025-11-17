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
# Used in StockDataset.MultiFeaturePriceDataset
# __init__(data, feature_cols, target_col="Close", seq_len=60, scaler=None)
DATASET_PARAMS = {
    "seq_len": [30, 50, 60, 90, 120],
    "scaler": list(SCALER_OPTIONS.keys())
}

# Model hyperparameters
# LSTM: matches model/LSTM.py signature
# __init__(input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.1, pkl_path=None)
LSTM_PARAMS = {
    "hidden_size": [32, 64, 128, 256],
    "num_layers": [1, 2, 3, 4],
    "output_size": [1],
    "dropout": [0.1, 0.2, 0.3, 0.4]
}

# GRU: matches model/GRU.py signature
# __init__(input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2, bidirectional=False, pkl_path=None)
GRU_PARAMS = {
    "hidden_size": [32, 64, 128, 256],
    "num_layers": [1, 2, 3, 4],
    "output_size": [1],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "bidirectional": [False, True]
}

# NBERT: matches model/NBERT.py signature  
# __init__(input_size, seq_len, output_size=1, hidden_dim=128, n_blocks=3, n_layers=4, dropout=0.1, pkl_path=None)
# Note: seq_len is passed separately in create_model()
NBERT_PARAMS = {
    "output_size": [1],
    "hidden_dim": [64, 128, 256],
    "n_blocks": [2, 3, 4, 5],
    "n_layers": [2, 3, 4, 5],
    "dropout": [0.1, 0.2, 0.3]
}

# Transformer: matches model/Transformer.py signature
# __init__(input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_size=1, pkl_path=None)
TRANSFORMER_PARAMS = {
    "d_model": [32, 64, 128],
    "nhead": [2, 4, 8],
    "num_layers": [1, 2, 3, 4],
    "dim_feedforward": [64, 128, 256, 512],
    "dropout": [0.1, 0.2, 0.3],
    "output_size": [1]
}

# Training hyperparameters
# Used in train_with_config() method
# optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])
# loss_fn = get_loss_function(training_params["loss_fn"])
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