import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Settings:
    tracking_uri: str
    experiment_names: Optional[List[str]]
    experiment_prefix: Optional[str]
    stock_tag_key: Optional[str]
    production_tag_key: str
    production_tag_value: str
    primary_metric: Optional[str]
    primary_metric_order: str
    model_artifact_path: str


def _parse_experiment_names(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return names or None


def load_settings() -> Settings:
    return Settings(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        experiment_names=_parse_experiment_names(os.getenv("MLFLOW_EXPERIMENT_NAMES")),
        experiment_prefix=os.getenv("MLFLOW_EXPERIMENT_PREFIX", "stock_") or None,
        stock_tag_key=os.getenv("MLFLOW_STOCK_TAG_KEY", "stock") or None,
        production_tag_key=os.getenv("MLFLOW_PRODUCTION_TAG_KEY", "production"),
        production_tag_value=os.getenv("MLFLOW_PRODUCTION_TAG_VALUE", "true"),
        primary_metric=os.getenv("MLFLOW_PRIMARY_METRIC") or None,
        primary_metric_order=os.getenv("MLFLOW_PRIMARY_METRIC_ORDER", "desc").lower(),
        model_artifact_path=os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "model"),
    )
