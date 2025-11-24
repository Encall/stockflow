import os
import tempfile
from pathlib import Path
from typing import List, Optional
import warnings

import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from drift_detector import DriftDetector
from minio_handler import MinioClient
from rolling_window_loader import RollingWindowDataLoader

warnings.filterwarnings(
    "ignore",
    message=".*ks_2samp: Exact calculation unsuccessful.*",
    category=RuntimeWarning,
)


def _env_get(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _require_storage_credentials() -> None:
    endpoint = _env_get("AWS_S3_ENDPOINT_URL", "MINIO_ENDPOINT")
    access = _env_get("AWS_ACCESS_KEY_ID", "MINIO_ACCESS_KEY")
    secret = _env_get("AWS_SECRET_ACCESS_KEY", "MINIO_SECRET_KEY")

    missing = []
    if not endpoint:
        missing.append("AWS_S3_ENDPOINT_URL")
    if not access:
        missing.append("AWS_ACCESS_KEY_ID")
    if not secret:
        missing.append("AWS_SECRET_ACCESS_KEY")

    if missing:
        raise SystemExit(
            "Missing required storage credentials. "
            "Set AWS_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY "
            "(or legacy MINIO_* variables)."
        )


def _parse_list_env(env_val: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated env var into a list, or None if empty."""
    if not env_val:
        return None
    parts = [p.strip() for p in env_val.split(",")]
    return [p for p in parts if p]


def _build_scaler(name: str | None):
    """Construct a scaler instance from a short name."""
    if not name or name.lower() == "none":
        return None
    name = name.lower()
    if name in {"standard", "standardscaler", "zscore"}:
        return StandardScaler()
    if name in {"minmax", "minmaxscaler"}:
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler '{name}'. Use standard|minmax|none.")


def _upload_files(paths: List[Path], remote_prefix: str, local_dir: Path) -> None:
    """Upload arbitrary report files to object storage and clean up."""
    if not paths:
        print("No report files configured for saving; skipping upload.", flush=True)
        return

    remote_prefix = remote_prefix.strip("/")
    client = MinioClient()
    for path in paths:
        if not path.exists():
            print(f"Report not found, skipping upload: {path}", flush=True)
            continue
        object_name = f"{remote_prefix}/{path.name}" if remote_prefix else path.name
        print(f"Uploading report '{path.name}' to storage at '{object_name}'...", flush=True)
        client.upload(path, object_name)
        try:
            path.unlink()
        except OSError as exc:
            print(f"Warning: could not delete temporary file {path}: {exc}", flush=True)

    try:
        local_dir.rmdir()
    except OSError:
        pass


def _extract_drift_flag(report: dict) -> bool:
    """Best-effort extraction of drift status from Evidently DataDriftPreset report."""
    try:
        metrics = report.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})
            if isinstance(result, dict):
                if result.get("dataset_drift") is not None:
                    return bool(result.get("dataset_drift"))
                if "drift_by_columns" in result:
                    drifted = any(col.get("drift_detected") for col in result["drift_by_columns"].values())
                    return bool(drifted)
    except Exception:
        return False
    return False

def _run_rolling_mode(remote_prefix: str, file_prefix: str, save_html: bool, save_json: bool) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="drift_reports_"))
    load_dotenv()
    _require_storage_credentials()

    stock_symbol = os.getenv("MONITORING_STOCK_SYMBOL")
    feature_cols = _parse_list_env(os.getenv("MONITORING_FEATURES"))
    feature_cols = [c for c in (feature_cols or []) if c]
    if not feature_cols:
        raise SystemExit("MONITORING_FEATURES must be set (comma-separated) to select monitoring columns.")
    target_col = os.getenv("MONITORING_TARGET", "close")
    scaler = _build_scaler(os.getenv("MONITORING_SCALER", "standard"))
    local_cache_dir = os.getenv("MONITORING_GOLD_CACHE", "data/gold")
    window_size = int(os.getenv("MONITORING_WINDOW_SIZE", "60"))
    split_size = int(os.getenv("MONITORING_SPLIT_SIZE", "30"))

    print("Starting DriftDetector (rolling window mode) with config:", flush=True)
    print(f"  save_dir (storage prefix): {remote_prefix}", flush=True)
    print(f"  file_prefix base: {file_prefix}", flush=True)
    print(f"  save_html: {save_html}", flush=True)
    print(f"  save_json: {save_json}", flush=True)
    print(f"  stock_symbol: {stock_symbol or 'ALL'}", flush=True)
    print(f"  feature_cols: {feature_cols or 'ALL'}", flush=True)
    print(f"  target_col: {target_col}", flush=True)
    print(f"  scaler: {scaler.__class__.__name__ if scaler else 'None'}", flush=True)
    print(f"  window_size/split_size: {window_size}/{split_size}", flush=True)

    loader = RollingWindowDataLoader(
        stock_symbol=stock_symbol or None,
        feature_cols=feature_cols,
        target_col=target_col,
        scaler=scaler,
        window_size=window_size,
        split_size=split_size,
        local_cache_dir=local_cache_dir,
    )

    try:
        df = loader.load()
        print(f"Loaded combined dataframe with {len(df)} rows.", flush=True)
    except Exception as exc:
        raise SystemExit(f"Failed to load gold parquet data: {exc}") from exc

    try:
        windows = list(loader.iter_windows(df))
    except Exception as exc:
        raise SystemExit(f"Failed to generate rolling windows: {exc}") from exc

    if not windows:
        raise SystemExit("No rolling windows generated.")

    print(f"Running drift for the most recent window only...", flush=True)
    window = windows[0]
    suffix = f"w{window.window_index}"
    if window.reference_period[0] and pd.notna(window.reference_period[0]):
        suffix = (
            f"{window.reference_period[0].date()}_"
            f"{window.current_period[1].date()}_w{window.window_index}"
        )
    window_prefix = f"{file_prefix}_{suffix}"
    detector = DriftDetector(
        save_dir=temp_dir,
        file_prefix=window_prefix,
        save_html=save_html,
        save_json=save_json,
        features=feature_cols,
    )
    # Ensure only monitoring features are passed to drift detector
    ref_df = window.reference.loc[:, feature_cols].copy()
    cur_df = window.current.loc[:, feature_cols].copy()
    print(
        f"  Window {window.window_index}: "
        f"ref {window.reference_period[0]} to {window.reference_period[1]} "
        f"current {window.current_period[0]} to {window.current_period[1]}",
        flush=True,
    )
    report = detector.check(ref_df, cur_df)

    drift_detected = _extract_drift_flag(report) if isinstance(report, dict) else False

    # Emit XCom-friendly JSON on stdout if requested (BashOperator do_xcom_push reads last line)
    if os.getenv("MONITORING_EMIT_XCOM", "false").lower() == "true":
        xcom_payload = {
            "drift_detected": drift_detected,
            "window_index": window.window_index,
            "reference_period_start": str(window.reference_period[0]),
            "reference_period_end": str(window.reference_period[1]),
            "current_period_start": str(window.current_period[0]),
            "current_period_end": str(window.current_period[1]),
            "report_prefix": window_prefix,
        }
        print(xcom_payload, flush=True)

    paths: List[Path] = []
    if save_html:
        paths.extend(temp_dir.glob("*.html"))
    if save_json:
        paths.extend(temp_dir.glob("*.json"))

    try:
        _upload_files(paths, remote_prefix, temp_dir)
    except Exception as e:
        raise SystemExit(f"Failed to upload drift reports to storage: {e}") from e

    print("Rolling drift check (single window) completed successfully and reports uploaded.", flush=True)


def main():
    # Read configuration from environment variables
    remote_prefix = os.environ.get("MONITORING_SAVE_DIR", "reports/data_drift")
    file_prefix = os.environ.get("MONITORING_FILE_PREFIX", "drift_report")
    save_html = os.environ.get("MONITORING_SAVE_HTML", "true").lower() == "true"
    save_json = os.environ.get("MONITORING_SAVE_JSON", "true").lower() == "true"
    _run_rolling_mode(remote_prefix, file_prefix, save_html, save_json)


if __name__ == "__main__":
    main()
