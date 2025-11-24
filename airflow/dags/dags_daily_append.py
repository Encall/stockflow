"""
Daily Stock Data Append Pipeline
Scrapes latest stock data daily and appends to existing datasets.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
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

    end = EmptyOperator(task_id="end")

    # Define task dependencies
    start >> bronze_append >> silver_append >> gold_append >> end
