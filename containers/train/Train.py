import torch
import model.LSTM as LSTMModel
import StockDataset
import GetDummies

def train_model(model, dataLoader, lr=0.001, epochs=30, loss_fn=torch.nn.MSELoss(), log_interval=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataLoader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % log_interval == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataLoader)}], Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(dataLoader)
        print(f'Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}')
    return model

def main():

    data = GetDummies.get_dummy(
        spec={
            "Open":"float",
            "High":"float",
            "Low":"float",
            "Close":"float",
            "Volume":"int"
        },
        n_rows=1000
    )
    feat_cols = ["Open", "High", "Low", "Volume"]
    target_col = ["Close"]

    stock_data = StockDataset.MultiFeaturePriceDataset(
        data=data,
        feature_cols=feat_cols,
        target_col=target_col,
        seq_len = 50
    )

    dataLoader = torch.utils.data.DataLoader(
        stock_data,
        batch_size=32,
        shuffle=True
    )

    model = LSTMModel.LSTM(
        input_size=len(feat_cols),
        hidden_size=64,
        num_layers=4,
        pkl_path=None
    )
    

    trained_model = train_model(
        model=model,
        dataLoader=dataLoader,
        lr=0.001,
        epochs=50,
        loss_fn=torch.nn.MSELoss(),
        log_interval=10
    )
    trained_model.save_model("lstm_stock_model.pkl")


if __name__ == "__main__":
    main()
    
