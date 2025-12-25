#!/usr/bin/env bash
set -euo pipefail

echo "🎯 Starting Model Pipeline Service..."
echo "whoami: $(whoami)"
echo "HOME: $HOME"
mkdir -p "$HOME/.cache" "$HOME/.ivy2" "$HOME/.config" "$SPARK_LOCAL_DIRS" /tmp/hadoop

# Extract AWS credentials from mounted file and export as environment variables
if [ -f "/aws/credentials" ]; then
    echo "🔑 Setting up AWS credentials for Spark S3A..."
    export AWS_ACCESS_KEY_ID=$(grep -A 10 "^\[default\]" /aws/credentials | grep "aws_access_key_id" | cut -d'=' -f2 | tr -d ' ')
    export AWS_SECRET_ACCESS_KEY=$(grep -A 10 "^\[default\]" /aws/credentials | grep "aws_secret_access_key" | cut -d'=' -f2 | tr -d ' ')
    echo "✅ AWS credentials extracted for Spark"
else
    echo "⚠️ AWS credentials file not found at /aws/credentials"
fi

echo "🤖 Pipeline: Model Training"
echo "☁️ S3 Bucket: ${S3_BUCKET}"
echo "📍 MLflow Tracking: ${MLFLOW_TRACKING_URI}"

# Wait for MLflow service to be ready
echo "⏳ Waiting for MLflow service..."
until curl -f "${MLFLOW_TRACKING_URI}/health" > /dev/null 2>&1; do
    echo "   MLflow not ready, waiting 5 seconds..."
    sleep 5
done
echo "✅ MLflow service is ready!"

# Configure Spark for S3A
export SPARK_CONF_DIR=/tmp/spark-conf
mkdir -p $SPARK_CONF_DIR

cat > $SPARK_CONF_DIR/spark-defaults.conf << EOF
spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain
spark.hadoop.fs.s3a.endpoint=s3.${AWS_REGION}.amazonaws.com
spark.hadoop.fs.s3a.path.style.access=false
spark.hadoop.fs.s3a.connection.ssl.enabled=true
spark.hadoop.fs.s3a.fast.upload=true
spark.hadoop.fs.s3a.multipart.size=67108864
spark.hadoop.fs.s3a.connection.timeout=60000
spark.hadoop.fs.s3a.socket.timeout=60000
# JARs are downloaded by PySpark automatically via spark.jars.packages
# No need to specify spark.jars path
EOF

echo "🔧 Spark S3A configuration completed"

# Fix Ivy cache and home directory permissions
export HOME=/tmp/home
export SPARK_LOCAL_DIRS=/tmp/spark
export IVY_CACHE_DIR=/tmp/ivy-cache
mkdir -p /tmp/home /tmp/ivy-cache /tmp/spark
chmod 777 /tmp/home /tmp/ivy-cache /tmp/spark

# Wait for data pipeline artifacts to be available (optional dependency)
echo "⏳ Checking for data artifacts in S3..."
python3 -c "
import boto3
import sys
from botocore.exceptions import NoCredentialsError, ClientError

try:
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket='${S3_BUCKET}', Prefix='artifacts/data_artifacts/', MaxKeys=1)
    if 'Contents' in response:
        print('✅ Data artifacts found in S3')
    else:
        print('⚠️ No data artifacts found - will use fallback data loading')
except Exception as e:
    print(f'⚠️ Could not check S3 data artifacts: {e}')
    print('   Continuing with training pipeline...')
"

# Run the model training pipeline
echo "🚀 Starting model training pipeline..."
exec python3 pipelines/training_pipeline.py
