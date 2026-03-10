# Production-Ready Machine Learning Pipeline System

A comprehensive, production-grade Machine Learning pipeline system designed for customer churn prediction. This project demonstrates best practices in MLOps, including end-to-end automation, reproducibility, and scalable deployment using Docker, Airflow, MLflow, and AWS services.

GitHub URL: https://github.com/p-sahas/customer_churn_prediction

## Features

* **End-to-End Pipelines**: Automated data processing, model training, and inference pipelines.
* **Infrastructure as Code**: Dockerized environment with `docker-compose` for consistent execution across different environments.
* **Workflow Orchestration**: Apache Airflow integration via DockerOperator for scheduled, reliable pipeline execution.
* **Experiment Tracking**: MLflow integration for tracking experiments, parameters, and metrics, utilizing an S3 backend for artifact storage.
* **Data Processing**:
  * Scalable processing capabilities using **PySpark** (embedded in containers).
  * Standard processing using **Pandas** and **Scikit-learn** for flexibility.
* **Streaming**: Integration with Confluent Kafka for real-time data ingestion scenarios.
* **Cloud Native Architecture**:
  * **AWS S3**: Centralized artifact storage for data, trained models, and logs.
  * **AWS RDS**: Managed relational database service for storing MLflow and Airflow metadata.
  * **AWS ECS**: Container orchestration for scalable deployment.

## Prerequisites

Before setting up the project, ensure you have the following installed:

* **Python**: Version 3.9 or higher.
* **Docker**: Docker Desktop or Docker Engine is required for containerized execution.
* **AWS Account**: Access to AWS S3 and RDS is required for full cloud integration.
* **Make**: GNU Make is used for running build and utility commands.

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd customer_churn_prediction
   ```
2. **Configure Environment**
   The project relies on environment variables for configuration. Copy the example environment file and update it with your specific credentials:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and provide values for:

   * `AWS_ACCESS_KEY_ID`
   * `AWS_SECRET_ACCESS_KEY`
   * `RDS_HOST`, `RDS_USER`, `RDS_PASSWORD` (if using RDS)
3. **Install Dependencies** (For Local Python Execution)
   It is recommended to use a virtual environment.

   ```bash
   make install
   ```

   This command creates a virtual environment in `.venv` and installs all required packages from `requirements.txt`.

## Usage

The project utilizes a `Makefile` to simplify common development and operations tasks.

### Local Python Execution

Run individual pipeline stages directly using your local Python environment.

* **Setup Directories**: Create necessary local data directories.
  ```bash
  make setup-dirs
  ```
* **Run Data Pipeline**: Execute the data preprocessing steps.
  ```bash
  make data-pipeline
  ```
* **Run Training Pipeline**: Train the machine learning model.
  ```bash
  make train-pipeline
  ```
* **Run Inference Pipeline**: Generate predictions using the trained model.
  ```bash
  make inference-pipeline
  ```
* **Run All**: Execute all three pipelines in sequence.
  ```bash
  make run-all
  ```

### Docker Execution (Recommended)

Deploy and run the full stack or individual components within Docker containers to ensure consistency.

* **Build Images**: Build all project Docker images, including the embedded PySpark support.
  ```bash
  make docker-build
  ```
* **Start Services**: Start MLflow tracking server and pipeline containers.
  ```bash
  make docker-up
  ```
* **Run All Pipelines**: Execute data, training, and inference pipelines sequentially inside Docker.
  ```bash
  make docker-run-all
  ```
* **Stop Services**: Stop all running containers.
  ```bash
  make docker-down
  ```
* **Clean Resources**: Remove project Docker resources.
  ```bash
  make docker-clean
  ```

### Airflow Automation

Manage scheduled workflows using Apache Airflow.

* **Initialize Airflow**: specificialize the Airflow database and create the admin user. This will clear existing DAG history.
  ```bash
  make airflow-init
  ```
* **Start Airflow**: Start the Airflow webserver and scheduler.
  ```bash
  make airflow-up
  ```
* **Start Full Automation**: Bring up both Docker pipeline services and Airflow orchestration.
  ```bash
  make automation-up
  ```

### Data Management & S3

Manage data and artifacts stored in AWS S3.

* **Upload Data**: Upload local `data/raw` and `data/processed` folders to S3 (one-time setup).
  ```bash
  make s3-upload-data
  ```
* **List Files**: List files in the S3 bucket with a specific prefix.
  ```bash
  make s3-list PREFIX=artifacts/
  ```
* **Clean S3 Artifacts**: Remove project artifacts (models, data) from S3.
  ```bash
  make s3-clean
  ```

## Project Structure

* `airflow/`: Contains Airflow DAG definitions and configuration settings.
* `artifacts/`: Directory for local storage of artifacts (if S3 is not configured).
* `data/`: Stores raw and processed datasets.
* `docker/`: Dockerfiles and related scripts for building container images.
* `pipelines/`: Core scripts for the machine learning pipelines (data processing, training, inference).
* `scripts/`: various utility scripts for AWS interaction, deployment, and setup.
* `src/`: Shared source code and python modules used across pipelines.
* `tests/`: Unit and integration tests to ensure code reliability.
* `.env`: Environment variables file (ensure this is not committed to version control).
* `Makefile`: Defines shortcuts for build, run, and utility commands.
* `docker-compose.yml`: Defines the services, networks, and volumes for the Docker setup.
* `requirements.txt`: List of Python dependencies required for the project.
* `setup.py`: Configuration for installing the project as a Python package.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

Credits

**Zuu Crew - ML Engineering Team**

Developed as part of the "Build Production-Ready Machine Learning Systems" course.
