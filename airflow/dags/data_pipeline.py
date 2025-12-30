import os
from airflow import DAG
from datetime import datetime, timedelta
from airflow.providers.docker.operators.docker import DockerOperator


# from airflow import DAG
# from airflow.utils import timezone
# from airflow.operators.python import PythonOperator
# from utils.airflow_tasks import validate_input_data, run_data_pipeline


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
                            image="churn-pipeline/data:latest"
                                )

    # # Step 1
    # validate_input_data_task = PythonOperator(
    #                                         task_id='validate_input_data',
    #                                         python_callable=validate_input_data,
    #                                         execution_timeout=timedelta(minutes=2)
    #                                         )

    # # Step 2
    # run_data_pipeline_task = PythonOperator(
    #                                         task_id='run_data_pipeline',
    #                                         python_callable=run_data_pipeline,
    #                                         execution_timeout=timedelta(minutes=15)
    #                                         )

    # validate_input_data_task >> run_data_pipeline_task