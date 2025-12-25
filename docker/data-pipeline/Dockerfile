# Data Pipeline Service  
FROM eclipse-temurin:17-jre

LABEL org.opencontainers.image.source="https://github.com/zuucrew/ml-pipeline"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="Data Pipeline with PySpark and S3 integration"

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create group/user and home dir with proper permissions
ARG UID=1001
ARG GID=1001
RUN addgroup --system --gid $GID app \
 && adduser --system --uid $UID --ingroup app --home /home/appuser --shell /bin/bash appuser \
 && mkdir -p /home/appuser /opt/app /tmp/spark /tmp/hadoop \
 && chown -R appuser:app /home/appuser /opt/app /tmp/spark /tmp/hadoop

# Set working directory
WORKDIR /opt/app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy project source code with proper ownership
COPY --chown=appuser:app src/ ./src/
COPY --chown=appuser:app utils/ ./utils/
COPY --chown=appuser:app pipelines/ ./pipelines/
COPY --chown=appuser:app config.yaml .

# Copy entrypoint script
COPY docker/data-pipeline/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER appuser

# Environment variables (consolidated)
ENV HOME=/home/appuser \
    SPARK_LOCAL_DIRS=/tmp/spark \
    HADOOP_OPTS="-Djava.io.tmpdir=/tmp/hadoop" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3 \
    MPLCONFIGDIR=/tmp/matplotlib \
    GIT_PYTHON_REFRESH=quiet

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=2 \
    CMD python3 -c "import boto3, os; boto3.client('s3').head_bucket(Bucket=os.environ.get('S3_BUCKET', 'zuucrew-mlflow-artifacts-prod'))" || exit 1

# Use entrypoint script
ENTRYPOINT ["bash", "/entrypoint.sh"]
