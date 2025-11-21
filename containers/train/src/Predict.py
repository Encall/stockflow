
from LoadTrainedModel import TrainedModel

if __name__ == "__main__":
    stock_name = "DIG"
    trained_model = TrainedModel(stock_name, model_version='latest', tracking_uri="http://127.0.0.1:5000")
    import pandas as pd
    sample_data = pd.DataFrame({
        "open": [100.0, 101.5, 102.0, 103.0, 104.5],
        "high": [101.0, 102.5, 103.0, 104.0, 105.5],
        "low": [99.5, 100.5, 101.0, 102.0, 103.5],
        "volume": [1500, 1600, 1700, 1800, 1900]
    })
    predictions = trained_model.predict(sample_data)
    print("Predictions:", predictions)