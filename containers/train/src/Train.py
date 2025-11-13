import torch
import model.LSTM as LSTMModel
import model.Transformer as TransformerModel
import model.GRU as GRUModel
import model.NBERT as NBERTModel
import StockDataset
import GetDummies
from sklearn.preprocessing import MinMaxScaler


def validate_model(model, data_loader, loss_fn=torch.nn.MSELoss()):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def test_model(model, data_loader, loss_fn=torch.nn.MSELoss()):
    """Evaluate model on test set and print loss."""
    avg_loss = validate_model(model, data_loader, loss_fn)
    print(f"Test Average Loss: {avg_loss:.4f}")
    return avg_loss


def train_model(
    model,
    train_loader,
    val_loader,
    lr=0.001,
    epochs=30,
    loss_fn=torch.nn.MSELoss(),
    log_interval=10,
):
    """Train model and validate every epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, loss_fn)

        print(
            f"Epoch {epoch + 1}/{epochs}   "
            f"Train Loss: {avg_train_loss:.4f}   "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    return model


def main():
    data = GetDummies.get_dummy(
        spec={
            "Open": "float",
            "High": "float",
            "Low": "float",
            "Close": "float",
            "Volume": "int",
        },
        n_rows=1500,
    )

    feat_cols = ["Open", "High", "Low", "Volume"]
    target_col = ["Close"]
    seq_len = 50

    stock_data = StockDataset.MultiFeaturePriceDataset(
        data=data,
        feature_cols=feat_cols,
        target_col=target_col,
        seq_len=seq_len,
        scaler=MinMaxScaler(),
    )

    train_ratio = 0.7
    val_ratio = 0.15

    train_size = int(train_ratio * len(stock_data))
    val_size = int(val_ratio * len(stock_data))
    test_size = len(stock_data) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        stock_data,
        [train_size, val_size, test_size],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    # LSTM
    # model = LSTMModel.LSTM(
    #     input_size=len(feat_cols),
    #     hidden_size=64,
    #     num_layers=4,
    #     pkl_path=None,
    # )

    # GRU
    model = GRUModel.GRU(
        input_size=len(feat_cols),
        hidden_size=64,
        num_layers=4,
        output_size=1,
        dropout=0.1,
        bidirectional=False,
        pkl_path=None,
    )

    # NBERT
    # model = NBERTModel.NBERT(
    #     input_size=len(feat_cols),
    #     seq_len=seq_len,
    #     output_size=1,
    #     dropout=0.1,
    #     hidden_dim=128,
    #     n_blocks=3,
    #     n_layers=4,
    #     pkl_path=None,
    # )

    # Transformer
    # model = TransformerModel.Transformer(
    #     input_size=len(feat_cols),   # 4 features
    #     d_model=64,
    #     nhead=4,
    #     num_layers=2,
    #     dim_feedforward=128,
    #     dropout=0.1,
    #     output_size=1,
    #     pkl_path=None,
    # )

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        epochs=20,
        loss_fn=torch.nn.MSELoss(),
        log_interval=20,
    )

    test_model(
        model=trained_model,
        data_loader=test_loader,
        loss_fn=torch.nn.MSELoss(),
    )

    # trained_model.save_model("stock_model.pkl")


if __name__ == "__main__":
    main()
