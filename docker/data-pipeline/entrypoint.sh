#!/usr/bin/env bash
set -euo pipefail

echo "🔄 Starting Data Pipeline Service..."
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
echo "📊 Pipeline: Data Preprocessing"
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

# Run the data pipeline
echo "🚀 Starting data preprocessing pipeline..."
exec python3 pipelines/data_pipeline.py
