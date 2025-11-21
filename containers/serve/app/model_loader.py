import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

from .config import Settings

logger = logging.getLogger(__name__)


class ProductionModelLoader:
    """Loads the latest MLflow run tagged for production and caches the pyfunc model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = MlflowClient(tracking_uri=settings.tracking_uri)
        mlflow.set_tracking_uri(settings.tracking_uri)
        self._lock = threading.Lock()
        # Cache per stock key (None = default)
        self._cache: Dict[Optional[str], Dict[str, Any]] = {}

    def _experiment_ids(self, stock: Optional[str]) -> List[str]:
        # If explicit experiments provided, use them (optionally filter by stock)
        if self.settings.experiment_names:
            ids = []
            for name in self.settings.experiment_names:
                candidate = name
                if stock:
                    if (
                        self.settings.experiment_prefix
                        and candidate != f"{self.settings.experiment_prefix}{stock}"
                    ):
                        # Skip non-matching experiments when stock is specified
                        continue
                exp = self._client.get_experiment_by_name(candidate)
                if exp:
                    ids.append(exp.experiment_id)
            if not ids:
                raise RuntimeError("No experiments found for provided names/stock")
            return ids

        # Otherwise derive experiment name from prefix + stock if provided
        if stock and self.settings.experiment_prefix:
            name = f"{self.settings.experiment_prefix}{stock}"
            exp = self._client.get_experiment_by_name(name)
            if not exp:
                raise RuntimeError(f"No experiment found for stock '{stock}'")
            return [exp.experiment_id]

        experiments = self._client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        ids = [exp.experiment_id for exp in experiments]
        if not ids:
            raise RuntimeError("No active MLflow experiments found")
        return ids

    def _order_by(self) -> List[str]:
        order = []
        if self.settings.primary_metric:
            direction = (
                "ASC" if self.settings.primary_metric_order.lower() == "asc" else "DESC"
            )
            order.append(f"metrics.{self.settings.primary_metric} {direction}")
        order.append("attributes.end_time DESC")
        return order

    def _find_production_run(
        self, stock: Optional[str]
    ) -> Tuple[str, str, Dict[str, Any]]:
        filters = [
            f"tags.{self.settings.production_tag_key} = "
            f"'{self.settings.production_tag_value}'"
        ]
        if stock and self.settings.stock_tag_key:
            filters.append(f"tags.{self.settings.stock_tag_key} = '{stock}'")
        filter_string = " and ".join(filters)

        runs = mlflow.search_runs(
            experiment_ids=self._experiment_ids(stock),
            filter_string=filter_string,
            order_by=self._order_by(),
            max_results=1,
        )

        if runs.empty:
            raise RuntimeError(
                f"No MLflow runs found with tag "
                f"{self.settings.production_tag_key}="
                f"{self.settings.production_tag_value}"
                + (f" and {self.settings.stock_tag_key}={stock}" if stock else "")
            )

        row = runs.iloc[0]
        run_id = row.run_id
        model_uri = f"runs:/{run_id}/{self.settings.model_artifact_path}"
        return run_id, model_uri, row.to_dict()

    def _load_model(self, stock: Optional[str], force: bool = False):
        cache_key = stock or "_default"
        cached = self._cache.get(cache_key)

        run_id, model_uri, run_row = self._find_production_run(stock)
        if (
            not force
            and cached is not None
            and cached["run_id"] == run_id
            and cached["model_uri"] == model_uri
        ):
            return

        logger.info(
            "Loading model for stock=%s from %s (run_id=%s)",
            stock,
            model_uri,
            run_id,
        )
        model = mlflow.pyfunc.load_model(model_uri)
        self._cache[cache_key] = {
            "model": model,
            "run_id": run_id,
            "model_uri": model_uri,
            "run_row": run_row,
        }

    def get_model(self, stock: Optional[str] = None):
        with self._lock:
            cache_key = stock or "_default"
            cached = self._cache.get(cache_key)
            if cached is None:
                self._load_model(stock)
                cached = self._cache.get(cache_key)
            return cached["model"]

    def refresh(self, stock: Optional[str] = None):
        with self._lock:
            self._load_model(stock, force=True)

    def model_info(self, stock: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            cache_key = stock or "_default"
            cached = self._cache.get(cache_key)
            if not cached:
                return {"run_id": None, "model_uri": None, "run_data": None}
            return {
                "run_id": cached["run_id"],
                "model_uri": cached["model_uri"],
                "run_data": cached["run_row"],
            }
