# Monitoring service

The monitoring app downloads gold-layer parquet data from an S3-compatible store (e.g., AWS S3 or MinIO), builds rolling 60-day windows (30-day reference + 30-day current), runs drift checks for each window, and uploads the resulting reports back to the bucket.

## Environment variables
- Storage: `AWS_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (optional), `AWS_S3_BUCKET` (default: `stockflow`), `AWS_S3_SECURE` (optional). Legacy `MINIO_*` equivalents are still recognized.
- Rolling input:
  - `MONITORING_STOCK_SYMBOL` (optional): limit to one symbol; leave empty to combine all gold parquets.
  - `MONITORING_FEATURES` (required): comma-separated list of feature columns to compare.
  - `MONITORING_TARGET` (default: `close`)
  - `MONITORING_SCALER` (default: `standard`; options: `standard|minmax|none`)
  - `MONITORING_GOLD_CACHE` (default: `data/gold`)
  - `MONITORING_WINDOW_SIZE` (default: `60`) and `MONITORING_SPLIT_SIZE` (default: `30`; must satisfy window_size == 2 * split_size)
- Reporting: `MONITORING_SAVE_DIR` (bucket prefix for reports), `MONITORING_FILE_PREFIX`, `MONITORING_SAVE_HTML`, `MONITORING_SAVE_JSON`

## Run
```bash
cd containers/monitoring
uv run python src/main.py
```

Reports for each window are saved under a unique file prefix (including window index/date range) in a temporary directory and then uploaded to `MONITORING_SAVE_DIR`. Temporary files are removed after upload. Ensure bucket credentials and gold parquet objects are available; the service will fail fast if required storage env vars are missing or no matching gold files are found.
