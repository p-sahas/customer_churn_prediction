#!/bin/bash
# ==========================================
# Local Deployment Script - Simplified
# ==========================================
# Starts Airflow (2 DAGs) and Kafka (3 containers)
#
# Usage: ./run_local.sh [options]
#
# Options:
#   --skip-build      Skip Docker image building (use existing images)
#   --stop            Stop all services
#   --status          Show status of all services
#   --logs            Follow logs of all services
#   -h, --help        Show this help message
#
# Prerequisites:
#   - Docker Desktop running
#   - AWS credentials configured (for S3)
#   - .env file with correct settings
# ==========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_BUILD=false
ACTION="start"

show_help() {
    cat << EOF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 LOCAL DEPLOYMENT - Airflow + Kafka
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage: ./run_local.sh [options]

Options:
  --skip-build      Skip Docker image building (use existing images)
  --stop            Stop all services
  --status          Show status of all services
  --logs            Follow logs of all services
  -h, --help        Show this help message

What it starts:
  • Airflow (2 DAGs): data_pipeline_every_20m, train_pipeline_every_60m
  • Kafka (3 containers): producer, inference, analytics
  • MLflow: Tracking server

Access URLs:
  • Airflow UI:  http://localhost:8080 (admin/admin)
  • MLflow UI:   http://localhost:5001
  • Kafka UI:    http://localhost:8090

Examples:
  ./run_local.sh                    # Start all services (build if needed)
  ./run_local.sh --skip-build       # Start without building
  ./run_local.sh --stop             # Stop all services
  ./run_local.sh --status           # Check status
  ./run_local.sh --logs             # View logs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --logs)
            ACTION="logs"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to show status
show_status() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 SERVICE STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # MLflow
    if docker ps | grep -q mlflow-tracking; then
        echo -e "   ${GREEN}✅ MLflow:${NC} Running at http://localhost:5001"
    else
        echo -e "   ${RED}❌ MLflow:${NC} Not running"
    fi
    
    # Airflow
    AIRFLOW_COUNT=$(docker ps | grep airflow | wc -l | tr -d ' ')
    if [ "$AIRFLOW_COUNT" -gt 0 ]; then
        echo -e "   ${GREEN}✅ Airflow:${NC} Running ($AIRFLOW_COUNT containers) at http://localhost:8080"
    else
        echo -e "   ${RED}❌ Airflow:${NC} Not running"
    fi
    
    # Kafka
    KAFKA_COUNT=$(docker ps | grep -E "kafka|zookeeper" | wc -l | tr -d ' ')
    if [ "$KAFKA_COUNT" -gt 0 ]; then
        echo -e "   ${GREEN}✅ Kafka:${NC} Running ($KAFKA_COUNT containers)"
        docker ps --format "      - {{.Names}}" | grep -E "kafka|zookeeper" || true
    else
        echo -e "   ${RED}❌ Kafka:${NC} Not running"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 Quick Commands:"
    echo "   View logs:     ./run_local.sh --logs"
    echo "   Stop all:      ./run_local.sh --stop"
    echo "   Restart:       ./run_local.sh"
    echo ""
}

# Function to stop services
stop_services() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🛑 STOPPING ALL SERVICES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "🛑 Stopping Kafka services..."
    docker-compose -f docker-compose.kafka.yml down 2>/dev/null || true
    echo ""
    
    echo "🛑 Stopping Airflow services..."
    docker-compose -f docker-compose.airflow.yml down 2>/dev/null || true
    echo ""
    
    echo "🛑 Stopping MLflow and pipelines..."
    docker-compose -f docker-compose.yml down 2>/dev/null || true
    echo ""
    
    echo -e "${GREEN}✅ All services stopped${NC}"
    echo ""
}

# Function to show logs
show_logs() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 VIEWING LOGS (Ctrl+C to exit)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Choose which logs to view:"
    echo ""
    echo "  AIRFLOW:"
    echo "    1) Airflow Scheduler"
    echo "    2) Airflow Webserver"
    echo "    3) Airflow Worker"
    echo "    4) Airflow Flower (monitoring)"
    echo ""
    echo "  KAFKA:"
    echo "    5) Kafka Producer"
    echo "    6) Kafka Consumer (churn_model.pkl loading)"
    echo "    7) Kafka Analytics"
    echo "    8) Kafka Broker"
    echo ""
    echo "  MLFLOW & DATABASES:"
    echo "    9) MLflow Tracking Server"
    echo "   10) Airflow PostgreSQL"
    echo "   11) Airflow Redis"
    echo ""
    echo "  COMBINED:"
    echo "   12) All Airflow services"
    echo "   13) All Kafka services"
    echo "   14) All services (everything)"
    echo ""
    read -p "Enter choice (1-14): " choice
    
    case $choice in
        1) docker logs airflow-scheduler -f --tail 100 ;;
        2) docker logs airflow-webserver -f --tail 100 ;;
        3) docker logs airflow-worker -f --tail 100 ;;
        4) docker logs airflow-flower -f --tail 100 ;;
        5) docker logs kafka-producer -f --tail 100 ;;
        6) docker logs kafka-inference -f --tail 100 ;;
        7) docker logs kafka-analytics -f --tail 100 ;;
        8) docker logs kafka-broker -f --tail 100 ;;
        9) docker logs mlflow-tracking -f --tail 100 ;;
        10) docker logs airflow-postgres -f --tail 100 ;;
        11) docker logs airflow-redis -f --tail 100 ;;
        12) docker-compose -f docker-compose.airflow.yml logs -f --tail 50 ;;
        13) docker-compose -f docker-compose.kafka.yml logs -f --tail 50 ;;
        14) echo "Showing all logs (last 30 lines each)..."
            docker-compose -f docker-compose.airflow.yml logs -f --tail 30 & \
            docker-compose -f docker-compose.kafka.yml logs -f --tail 30 & \
            docker logs mlflow-tracking -f --tail 30 & \
            wait ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
}

# Handle different actions
if [ "$ACTION" = "status" ]; then
    show_status
    exit 0
fi

if [ "$ACTION" = "stop" ]; then
    stop_services
    exit 0
fi

if [ "$ACTION" = "logs" ]; then
    show_logs
    exit 0
fi

# Main deployment starts here
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 LOCAL DEPLOYMENT - Airflow + Kafka"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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

# Clean up existing containers and ports
echo "🧹 Cleaning up existing containers..."
# Stop Homebrew Kafka if running
brew services stop kafka >/dev/null 2>&1 || true
# Remove Docker containers
docker ps -a | grep -E "kafka|airflow|mlflow" | awk '{print $1}' | xargs -r docker rm -f >/dev/null 2>&1 || true
# Kill processes on ports
lsof -ti:9092,9093,9094,29092,8080,5555,8793 2>/dev/null | xargs kill -9 2>/dev/null || true
echo -e "${GREEN}✅ Cleanup complete${NC}"
echo ""

# Check AWS credentials
echo "🔍 Checking AWS credentials..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  AWS credentials not found${NC}"
    echo "   S3 uploads may fail. Configure with: aws configure"
else
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo -e "${GREEN}✅ AWS credentials OK (Account: $ACCOUNT_ID)${NC}"
fi
echo ""

# Check .env file
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "   Copy .env.example to .env and configure it"
    exit 1
fi
echo -e "${GREEN}✅ .env file found${NC}"
echo ""

# Build images if needed
if [ "$SKIP_BUILD" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 CHECKING DOCKER IMAGES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Check pipeline images
    IMAGES_MISSING=false
    for img in churn-pipeline/data:latest churn-pipeline/model:latest; do
        if ! docker image inspect "$img" >/dev/null 2>&1; then
            IMAGES_MISSING=true
            break
        fi
    done
    
    if [ "$IMAGES_MISSING" = true ]; then
        echo "📦 Building pipeline images..."
        docker-compose -f docker-compose.yml build
        echo -e "${GREEN}✅ Pipeline images built${NC}"
    else
        echo -e "${GREEN}✅ Pipeline images exist${NC}"
    fi
    echo ""
    
    # Check Airflow image
    if ! docker image inspect churn-pipeline/airflow:2.8.1-amazon >/dev/null 2>&1; then
        echo "📦 Building Airflow image..."
        docker build -t churn-pipeline/airflow:2.8.1-amazon -f docker/Dockerfile.airflow .
        echo -e "${GREEN}✅ Airflow image built${NC}"
    else
        echo -e "${GREEN}✅ Airflow image exists${NC}"
    fi
    echo ""
    
    # Check Kafka images
    KAFKA_MISSING=false
    for img in kafka/churn-pipeline-producer:latest kafka/churn-pipeline-inference:latest kafka/churn-pipeline-analytics:latest; do
        if ! docker image inspect "$img" >/dev/null 2>&1; then
            KAFKA_MISSING=true
            break
        fi
    done
    
    if [ "$KAFKA_MISSING" = true ]; then
        echo "📦 Building Kafka images..."
        docker-compose -f docker-compose.kafka.yml build
        echo -e "${GREEN}✅ Kafka images built${NC}"
    else
        echo -e "${GREEN}✅ Kafka images exist${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}⚠️  Skipping image build (using existing images)${NC}"
    echo ""
fi

# Start services
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STARTING SERVICES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Start MLflow
echo "1️⃣ Starting MLflow..."
docker-compose -f docker-compose.yml up -d mlflow-tracking
echo -e "${GREEN}✅ MLflow started${NC}"
echo ""

# Wait for MLflow
echo "⏳ Waiting for MLflow to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:5001/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ MLflow is ready${NC}"
        break
    fi
    sleep 2
done
echo ""

# 2. Start Airflow
echo "2️⃣ Starting Airflow..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker-compose -f docker-compose.airflow.yml up -d --no-build
echo -e "${GREEN}✅ Airflow started${NC}"
echo ""

# Wait for Airflow
echo "⏳ Waiting for Airflow to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Airflow is ready${NC}"
        break
    fi
    sleep 3
done
echo ""

# 3. Start Kafka
echo "3️⃣ Starting Kafka services..."
docker-compose -f docker-compose.kafka.yml up -d
echo -e "${GREEN}✅ Kafka services started${NC}"
echo ""

# Wait for Kafka
echo "⏳ Waiting for Kafka to be ready..."
sleep 15
echo -e "${GREEN}✅ Kafka is ready${NC}"
echo ""

# Final status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ LOCAL DEPLOYMENT COMPLETE!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Access URLs:"
echo "   • Airflow UI:  http://localhost:8080 (admin/admin)"
echo "   • MLflow UI:   http://localhost:5001"
echo "   • Kafka UI:    http://localhost:8090"
echo ""
echo "📋 Airflow DAGs (Unified Timestamp):"
echo "   • data_pipeline_every_20m   → Runs every 20 min"
echo "   • train_pipeline_every_60m  → Runs every 60 min"
echo ""
echo "🔄 Kafka Services:"
echo "   • Producer:   Streaming customer events"
echo "   • Consumer:   Real-time inference with churn_model.pkl"
echo "   • Analytics:  Aggregating to PostgreSQL"
echo ""
echo "💡 Useful commands:"
echo "   • Check status:  ./run_local.sh --status"
echo "   • View logs:     ./run_local.sh --logs"
echo "   • Stop all:      ./run_local.sh --stop"
echo ""
echo "📊 View logs individually:"
echo "   • docker logs airflow-scheduler -f"
echo "   • docker logs kafka-inference -f"
echo "   • docker logs kafka-producer -f"
echo "   • docker logs kafka-analytics -f"
echo ""
echo -e "${GREEN}🎉 All services are running!${NC}"
echo ""