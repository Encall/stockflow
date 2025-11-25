import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from app.config import Settings
from app.model_loader import ProductionModelLoader


class ProductionModelLoaderTests(unittest.TestCase):
    @mock.patch("app.model_loader.mlflow")
    @mock.patch("app.model_loader.MlflowClient")
    def test_loads_production_tagged_run_once(self, client_mock, mlflow_mock):
        client_mock.return_value.search_experiments.return_value = [
            SimpleNamespace(experiment_id="1")
        ]
        mlflow_mock.search_runs.return_value = pd.DataFrame(
            [
                {
                    "run_id": "run-123",
                    "tags.production": "true",
                    "attributes.end_time": 123,
                }
            ]
        )
        mlflow_mock.pyfunc.load_model.return_value = "model"

        settings = Settings(
            tracking_uri="http://mlflow:5000",
            experiment_names=None,
            experiment_prefix="stock_",
            stock_tag_key="stock",
            production_tag_key="production",
            production_tag_value="true",
            primary_metric=None,
            primary_metric_order="desc",
            model_artifact_path="model",
        )
        loader = ProductionModelLoader(settings)

        first_model = loader.get_model()
        second_model = loader.get_model()

        self.assertEqual(first_model, "model")
        self.assertIs(second_model, first_model)
        mlflow_mock.pyfunc.load_model.assert_called_once_with(
            "runs:/run-123/model"
        )
        mlflow_mock.search_runs.assert_called_once()

    @mock.patch("app.model_loader.mlflow")
    @mock.patch("app.model_loader.MlflowClient")
    def test_loads_by_stock_experiment_and_tag(self, client_mock, mlflow_mock):
        client_mock.return_value.get_experiment_by_name.return_value = SimpleNamespace(
            experiment_id="dig-exp"
        )
        mlflow_mock.search_runs.return_value = pd.DataFrame(
            [
                {
                    "run_id": "run-dig",
                    "tags.production": "true",
                    "tags.stock": "DIG",
                    "attributes.end_time": 200,
                }
            ]
        )
        mlflow_mock.pyfunc.load_model.return_value = "dig-model"

        settings = Settings(
            tracking_uri="http://mlflow:5000",
            experiment_names=None,
            experiment_prefix="stock_",
            stock_tag_key="stock",
            production_tag_key="production",
            production_tag_value="true",
            primary_metric=None,
            primary_metric_order="desc",
            model_artifact_path="model",
        )
        loader = ProductionModelLoader(settings)

        model = loader.get_model(stock="DIG")

        self.assertEqual(model, "dig-model")
        mlflow_mock.search_runs.assert_called_once()
        args, kwargs = mlflow_mock.search_runs.call_args
        self.assertIn("tags.stock = 'DIG'", kwargs["filter_string"])


if __name__ == "__main__":
    unittest.main()
