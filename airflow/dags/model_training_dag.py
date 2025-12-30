"""
Model Training DAG - Daily (Every 24 hours)
Uses DockerOperator to run churn-pipeline/model:latest container
"""

import os
import platform
from airflow import DAG
from datetime import datetime, timedelta
from airflow.providers.docker.operators.docker import DockerOperator

# Cross-platform Docker socket configuration
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

default_args = {
    "owner": "ml_engineering_team",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "start_date": datetime(2025, 12, 30),
    "catchup": False
}

with DAG(
    dag_id="train_pipeline_daily",
    default_args=default_args,
    schedule="0 0 * * *",  # Every day at midnight (24 hours)
    max_active_runs=1,  # Prevent overlap
    dagrun_timeout=timedelta(hours=2),
    description="Run model training every 24h via DockerOperator with unified timestamp and joblib",
    tags=["ml_pipeline", "model_training", "sklearn", "mlflow", "docker", "s3", "joblib"]
) as dag:

    run_training_pipeline = DockerOperator(
        task_id="run_training_pipeline",
        image="churn-pipeline/model:latest",
        api_version="auto",
        auto_remove=True,
        docker_url=DOCKER_URL,  # Cross-platform compatible
        network_mode="churn-pipeline-network",
        mount_tmp_dir=False,  # Disable tmp mount for macOS Docker Desktop compatibility
        environment={
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            "S3_BUCKET": os.getenv("S3_BUCKET"),
            "MLFLOW_TRACKING_URI": "http://mlflow-tracking:5001",
            "CONTAINERIZED": "true",
            "USE_S3": "true",
            "SKIP_S3_UPLOAD": "false"
        },
        # Container uses entrypoint, so no command needed
        execution_timeout=timedelta(hours=2),
    )

# Task explanation:
# - Runs churn-pipeline/model:latest container via DockerOperator
# - Executes training_pipeline.py with sklearn engine
# - REUSES latest data timestamp for lineage (unified timestamp)
# - Uploads artifacts to S3: artifacts/train/<TIMESTAMP>/
# - Logs to MLflow with both data and train S3 paths
# - Environment variables:
#   - USE_S3=true: Force S3 usage
#   - SKIP_S3_UPLOAD=false: Ensure uploads happen
#   - MLFLOW_TRACKING_URI: MLflow server for tracking
# - Artifacts: churn_model.pkl (joblib), metadata, metrics, params, feature_importance
# - Model format: joblib with compress=3 (74% smaller than pickle)
# - Schedule: Daily (0 0 * * *) - matches ECS deployment
