import os
import platform
from airflow import DAG
from datetime import datetime, timedelta
from airflow.providers.docker.operators.docker import DockerOperator

def get_docker_url():
    """
    Returns the appropriate Docker socket URL based on the platform.
    
    Returns:
        str: Docker socket URL
            - Linux/macOS: unix://var/run/docker.sock
            - Windows: npipe:////./pipe/docker_engine
    """
    system = platform.system()
    if system == "Windows":
        return "npipe:////./pipe/docker_engine"
    else:
        # Linux, macOS, WSL
        return "unix://var/run/docker.sock"

DOCKER_URL = get_docker_url()

default_arguments = {
                    'owner' : 'sahas',
                    'depends_on_past' : False,
                    'retries': 2,
                    'retry_delay': timedelta(minutes=2),
                    'start_date': datetime(2025, 12,31),
                    'catchup' : False,
                    'email_on_failuer': False,
                    'email_on_retry': False,
                    }

with DAG(
        dag_id = 'data_pipeline_evey_20m',
        default_args = default_arguments,
        schedule_interval='*/20 * * * *', # Every 20 min
        catchup=False,
        max_active_runs=1,
        dagrun_timeout=timedelta(minutes=30),
        description='Data Pipeline Local - DockerOperater',
        tags=['pyspark', 'mllib', 'mlflow', 'batch-processing']
        ) as dag:

    run_data_pipeline = DockerOperator(
                            task_id="run_data_pipeline",
                            image="churn-pipeline/data:latest",
                            api_version='auto', auto_remove=True,
                            docker_url='s',
                            network_model='churn-pipeline-network',
                            mount_tmp_dir=False,
                            environment={
                                        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
                                        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
                                        "AWS_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                                        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                                        "S3_BUCKET": os.getenv("S3_BUCKET"),
                                        "MLFLOW_TRACKING_URI": "http://mlflow-tracking:5001",
                                        "CONTAINERIZED": "true"
                                        },
                            # Container uses entrypoint, so no command needed
                            execution_timeout=timedelta(minutes=30),  
                                                    )
