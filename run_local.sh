#!/bin/bash
# ==========================================
# Local Development Environment Setup
# ==========================================
# Complete end-to-end deployment of local ML pipeline
#
# Usage: ./run_local.sh
#
# Prerequisites:
#   - Docker and Docker Compose installed
#   - AWS credentials configured (for S3 and RDS access)
#   - .env file with correct settings
# ==========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set environment variables
export MLFLOW_TRACKING_URI="http://localhost:5001"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://customer-churn-prediction-sahas-zuucrew/artifacts/mlflow-artifacts"
export PYTHONPATH="."
export AWS_PROFILE="${AWS_PROFILE:-default}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 LOCAL DEPLOYMENT - End to End"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check AWS credentials
echo "🔍 Checking AWS credentials..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo -e "${RED}❌ AWS credentials not found or invalid!${NC}"
    echo ""
    echo "AWS credentials are needed for S3 access and RDS."
    echo ""
    echo "Please configure AWS credentials:"
    echo "  1. Run: aws configure"
    echo "  2. Or set: export AWS_PROFILE=default"
    echo "  3. Or check: ~/.aws/credentials"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region 2>/dev/null || echo "ap-south-1")

echo -e "${GREEN}✅ AWS credentials verified${NC}"
echo "   Account: $ACCOUNT_ID"
echo "   Region: $REGION"
echo "   Profile: $AWS_PROFILE"
echo ""

# Check Docker
echo "🐳 Checking Docker..."
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running!${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"
echo ""

# Confirmation prompt
echo -e "${YELLOW}This will start the complete local ML pipeline:${NC}"
echo "  • MLflow Tracking Server"
echo "  • Airflow (webserver, scheduler, worker, flower)"
echo "  • Local PostgreSQL (for Airflow metadata)"
echo "  • Redis (for Celery backend)"
echo "  • 3 ML Pipeline containers (data, model, inference)"
echo ""
read -p "Continue with local deployment? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo -e "${RED}❌ Deployment cancelled${NC}"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Deployment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Build Docker images
echo -e "${BLUE}1️⃣ Building Docker images...${NC}"
make docker-build
echo ""

# Step 2: Build Airflow image
echo -e "${BLUE}2️⃣ Building Airflow image...${NC}"
make airflow-build
echo ""

# Step 3: Start MLflow and pipeline services
echo -e "${BLUE}3️⃣ Starting MLflow and pipeline services...${NC}"
make docker-up
echo ""

# Wait for MLflow to be ready
echo "⏳ Waiting for MLflow to initialize..."
sleep 10
echo ""

# Step 4: Initialize Airflow with local PostgreSQL
echo -e "${BLUE}4️⃣ Initializing Airflow (local PostgreSQL)...${NC}"
echo -e "${YELLOW}⚠️  This will clear ALL DAG history and create a fresh database${NC}"
make airflow-init
echo ""

# Step 5: Start Airflow services
echo -e "${BLUE}5️⃣ Starting Airflow services...${NC}"
make airflow-up
echo ""

# Wait for Airflow to be ready
echo "⏳ Waiting for Airflow to initialize..."
sleep 15
echo ""

# Verify services
echo "🔍 Verifying services..."
echo ""

# Check Docker containers
RUNNING_CONTAINERS=$(docker ps --filter "status=running" | grep -c "churn-pipeline\|airflow\|mlflow" || echo "0")
echo "   Running containers: $RUNNING_CONTAINERS"

# Check Airflow health
if docker exec airflow-webserver airflow db check >/dev/null 2>&1; then
    echo -e "   ${GREEN}✅ Airflow database: Connected${NC}"
else
    echo -e "   ${YELLOW}⚠️  Airflow database: Check manually${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ LOCAL DEPLOYMENT COMPLETE!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Access URLs:"
echo "   • Airflow UI: http://localhost:8080 (admin/admin)"
echo "   • MLflow UI:  http://localhost:5001"
echo "   • Flower UI:  http://localhost:5555"
echo ""
echo "📋 Available DAGs:"
echo "   • data_pipeline_every_20m      → Every 20 minutes"
echo "   • inference_pipeline_every_10m → Every 10 minutes"
echo "   • train_pipeline_every_60m     → Every 60 minutes (hourly)"
echo ""
echo "💡 Useful commands:"
echo "   • Check status:  make docker-status && make airflow-status"
echo "   • View logs:     docker logs airflow-scheduler -f"
echo "   • Stop all:      make airflow-down && make docker-down"
echo "   • Full cleanup:  make clean-all"
echo ""
echo -e "${GREEN}🎉 Local environment is ready!${NC}"
echo ""