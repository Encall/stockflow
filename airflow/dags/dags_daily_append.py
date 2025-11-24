"""
Daily Stock Data Append Pipeline
Scrapes latest stock data daily and appends to existing datasets.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
import json
import logging
import airflow

def get_minio_config():
    """Retrieves MinIO configuration from Airflow Variables."""
    return {
        'AWS_ACCESS_KEY_ID': airflow.models.Variable.get("minio_access_key"),
        'AWS_SECRET_ACCESS_KEY': airflow.models.Variable.get("minio_secret_key"),
        'AWS_S3_ENDPOINT_URL': airflow.models.Variable.get("minio_endpoint"),
        'AWS_REGION': airflow.models.Variable.get("minio_region"),
        'STOCK_TICKERS': airflow.models.Variable.get("stock_tickers", default_var="AAPL,GOOGL,MSFT"),
    }


def get_monitoring_config():
    """Return monitoring-specific environment variables.

    These are intentionally separate from the shared MinIO config so other
    containers don't receive monitoring-only settings.
    """
    return {
        'MONITORING_STOCK_SYMBOL': airflow.models.Variable.get("monitoring_stock_symbol", default_var=""),
        'MONITORING_FEATURES': airflow.models.Variable.get("monitoring_features", default_var="open,high,low,volume"),
        'MONITORING_TARGET': airflow.models.Variable.get("monitoring_target", default_var="close"),
        'MONITORING_SCALER': airflow.models.Variable.get("monitoring_scaler", default_var="standard"),
        'MONITORING_GOLD_CACHE': airflow.models.Variable.get("monitoring_gold_cache", default_var="data/gold"),
        'MONITORING_WINDOW_SIZE': airflow.models.Variable.get("monitoring_window_size", default_var="60"),
        'MONITORING_SPLIT_SIZE': airflow.models.Variable.get("monitoring_split_size", default_var="30"),
        'MONITORING_SAVE_DIR': airflow.models.Variable.get("monitoring_save_dir", default_var="reports/data_drift"),
        'MONITORING_FILE_PREFIX': airflow.models.Variable.get("monitoring_file_prefix", default_var="drift_report"),
        'MONITORING_SAVE_HTML': airflow.models.Variable.get("monitoring_save_html", default_var="true"),
        'MONITORING_SAVE_JSON': airflow.models.Variable.get("monitoring_save_json", default_var="true"),
        'MONITORING_EMIT_XCOM': airflow.models.Variable.get("monitoring_emit_xcom", default_var="true"),
    }

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 11, 22),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="stockflow_daily_append",
    default_args=default_args,
    description="Daily incremental stock data scraping and processing",
    schedule="0 18 * * 1-5",  # 6 PM every weekday (after market close)
    catchup=False,  # Don't backfill historical runs
    max_active_runs=1,  # Only one run at a time
    tags=["stockflow", "daily", "incremental"],
) as dag:

    start = EmptyOperator(task_id="start")

    # Scrape latest data from yfinance and append to bronze layer
    bronze_append = DockerOperator(
        task_id="bronze_append",
        image="ghcr.io/encall/stockflow/etl:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        force_pull=False,  # Set to True if you want to always pull latest image
        command="bronze-append",
    )

    # Clean and append to silver layer
    silver_append = DockerOperator(
        task_id="silver_append",
        image="ghcr.io/encall/stockflow/etl:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        force_pull=False,
        command="silver-append",
    )

    # Create features and append to gold layer
    gold_append = DockerOperator(
        task_id="gold_append",
        image="ghcr.io/encall/stockflow/etl:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        force_pull=False,
        command="gold-append",
    )

    # Monitoring needs both MinIO credentials and monitoring-specific options.
    # Merge the two small dicts so only the monitoring task receives monitoring-only envs.
    monitoring = DockerOperator(
        task_id="run_monitoring",
        image="ghcr.io/encall/stockflow/monitoring:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment={**get_minio_config(), **get_monitoring_config()},
        force_pull=False,
        # Only push the last stdout line to XCom (compact JSON payload emitted
        # by the monitoring container as its final line).
        xcom_all=False,
    )

    def consume_monitoring_output(**context):
        """Pull XCom emitted by the monitoring container and log it for testing.

        The monitoring container emits a JSON payload to stdout when
        `MONITORING_EMIT_XCOM=true`. This function tries to parse and
        log the payload so we can verify XCom propagation in Airflow.
        """
        ti = context['ti']
        payload = ti.xcom_pull(task_ids='run_monitoring')
        logging.info('Raw XCom payload from run_monitoring: %s', payload)

        # Helper: try parse a single string as JSON, then as Python literal.
        def _try_parse_string(s: str):
            s = s.strip()
            try:
                return json.loads(s)
            except Exception:
                pass
            try:
                import ast

                return ast.literal_eval(s)
            except Exception:
                return None

        parsed = None
        # DockerOperator with xcom_all=True often returns a list of log lines.
        if isinstance(payload, list):
            # Find the last log line that looks like a dict/JSON and parse it.
            for line in reversed(payload):
                if not isinstance(line, str):
                    continue
                if "drift_detected" in line or line.strip().startswith("{"):
                    parsed = _try_parse_string(line)
                    if isinstance(parsed, dict):
                        break

        elif isinstance(payload, str):
            parsed = _try_parse_string(payload)

        # If we parsed a dict, return a compact selection so XCom stays small.
        if isinstance(parsed, dict):
            keys = [
                "drift_detected",
                "window_index",
                "report_prefix",
                "reference_period_start",
                "reference_period_end",
                "current_period_start",
                "current_period_end",
            ]
            reduced = {k: parsed.get(k) for k in keys if k in parsed}
            logging.info('Selected XCom payload extracted: %s', reduced)
            return reduced

        logging.warning('No structured XCom payload found; returning raw payload')
        return payload

    consume_xcom = PythonOperator(
        task_id='consume_monitoring_xcom',
        python_callable=consume_monitoring_output,
    )

    end = EmptyOperator(task_id="end")

    # Define task dependencies
    start >> bronze_append >> silver_append >> gold_append >> monitoring >> consume_xcom >> end
