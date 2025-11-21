# Model Serving

Serves the MLflow run tagged for production via FastAPI.

## Env
- `MLFLOW_TRACKING_URI` (default `http://mlflow:5000`)
- `MLFLOW_EXPERIMENT_NAMES` (comma-separated; optional, defaults to all experiments)
- `MLFLOW_EXPERIMENT_PREFIX` (default `stock_`; used when routing by stock name)
- `MLFLOW_STOCK_TAG_KEY` (default `stock`; added to filter runs when stock is provided)
- `MLFLOW_PRODUCTION_TAG_KEY` / `MLFLOW_PRODUCTION_TAG_VALUE` (defaults `production=true`)
- `MLFLOW_MODEL_ARTIFACT_PATH` (default `model`, the logged pyfunc path)
- `MLFLOW_PRIMARY_METRIC` + `MLFLOW_PRIMARY_METRIC_ORDER` (`desc`/`asc`; optional ordering hint)
- S3/MinIO credentials: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT_URL`, `MLFLOW_S3_ENDPOINT_URL`

## Run
```bash
cp containers/serve/.env.example .env  # adjust values
docker compose up model-serving
```

## API
- `GET /health` – status and loaded run info
- `GET /metadata` – run metadata (optional `stock` query)
- `POST /predict` – body: `{"instances": [{...feature dict...}], "stock": "DIG"}`; returns predictions plus run_id/model_uri. `stock` selects the experiment and filters by `MLFLOW_STOCK_TAG_KEY`.
- `POST /reload` – forces reload of the latest production-tagged run (optional `stock` query)

Tag a run in MLflow with `production=true` (or the configured tag) so it gets selected. Among tagged runs the service orders by the configured primary metric (if set) then by latest end time.
