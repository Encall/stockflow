import mlflow
from mlflow.tracking import MlflowClient
import os


class TrainedModel():
    def __init__(self, stock_name: str, model_version='latest', tracking_uri="http://127.0.0.1:5000"):
        """Initializes the TrainedModel by loading from MLflow."""
        self.stock_name = stock_name
        self.model_version = model_version
        self.tracking_uri = tracking_uri
        self.model = self._load_model_from_mlflow()

    def _load_model_from_mlflow(self):
        """Loads the trained model from MLflow Model Registry."""
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()
        model_name = f"{self.stock_name}_model"
        try:
            if self.model_version == 'latest':
                model_version_info = client.get_latest_versions(model_name, stages=["Production"])
                if not model_version_info:
                    raise ValueError(f"No production model found for {model_name}")
                model_version = model_version_info[0].version
            else:
                model_version = self.model_version

            model_uri = f"models:/{model_name}/{model_version}"
            print(f"Loading model from MLflow URI: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model {model_name} version {model_version} loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            return None
        
    def predict(self, input_data):
        """Makes predictions using the loaded model."""
        if self.model is None:
            raise ValueError("Model is not loaded.")
        return self.model.predict(input_data)
        