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
        'AWS_ACCESS_KEY_ID': airflow.models.Variable.get("minio_access_key"),
        'AWS_SECRET_ACCESS_KEY': airflow.models.Variable.get("minio_secret_key"),
        'AWS_S3_ENDPOINT_URL': airflow.models.Variable.get("minio_endpoint"),
        'AWS_REGION': airflow.models.Variable.get("minio_region"),
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
    tags=["stockflow", "trigger_train"],
) as dag:

    start = EmptyOperator(task_id="start")

    # pull container image from ghcr.io and run the process
    bronze_container = DockerOperator(
        task_id="run_etl_bronze_layer",
        image="ghcr.io/encall/stockflow/etl:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        force_pull=True,
        command=["bronze"]
    )
    
    silver_container = DockerOperator(
        task_id="run_etl_silver_layer",
        image="ghcr.io/encall/stockflow/etl:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        force_pull=True,
        command=["silver"]
    )
    
    gold_container = DockerOperator(
        task_id="run_etl_gold_layer",
        image="ghcr.io/encall/stockflow/etl:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        force_pull=True,
        command=["gold"]
    )
    
    train_container = DockerOperator(
        task_id="run_train_container",
        image="ghcr.io/encall/stockflow/train:latest",
        api_version="auto",
        auto_remove="success",
        tty=True,
        docker_url="unix://var/run/docker.sock",
        environment=get_minio_config(),
        force_pull=True,
        command=["--exhaustive"]
    )

    end = EmptyOperator(task_id="end")

    start >> bronze_container >> silver_container >> gold_container >> train_container >> end