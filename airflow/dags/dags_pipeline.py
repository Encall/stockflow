from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
from pathlib import Path
import airflow

# ฟังก์ชันจะรันตอน task execution
def get_minio_config():
    return {
        'MINIO_ACCESS_KEY': airflow.models.Variable.get("minio_access_key"),
        'MINIO_SECRET_KEY': airflow.models.Variable.get("minio_secret_key"),
        'MINIO_ENDPOINT': airflow.models.Variable.get("minio_endpoint"),
        'MINIO_REGION': airflow.models.Variable.get("minio_region"),
    }

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 2, 26),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "stockflow",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["stockflow"],
) as dag:

    start = EmptyOperator(task_id="start")

    # pull container image from ghcr.io and run the process
    silver_container = DockerOperator(
        task_id="run_etl_container",
        image="ghcr.io/encall/stockflow/etl:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        command=["silver"]
    )
    gold_container = DockerOperator(
        task_id="run_gold_container",
        image="ghcr.io/encall/stockflow/etl:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        command=["gold"]
    )

    end = EmptyOperator(task_id="end")

    start >> silver_container >> gold_container >> end