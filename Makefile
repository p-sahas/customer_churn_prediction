.ONESHELL:

.PHONY: all clean install setup-dirs train-pipeline data-pipeline streaming-inference run-all help re-run-all

# Default Python interpreter
PYTHON = python
VENV = .venv/bin/activate

# Activate venv
#source .venv/bin/activate

MLFLOW_PORT ?= 5001

SHELL := /bin/bash

# Default target
all: help

# Help target
help:
	@echo "🚀 Production ML Pipeline System"
	@echo "================================="
	@echo ""
	@echo "📦 Setup Commands:"
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make setup-dirs          - Create necessary directories for pipelines"
	@echo "  make clean               - Clean up local cache only (safe)"
	@echo "  make clean-local-artifacts - Remove legacy local artifact folders"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make docker-build        - Build all Docker images"
	@echo "  make docker-up           - Start all Docker services"
	@echo "  make docker-down         - Stop all Docker services"
	@echo "  make docker-data-pipeline    - Run data pipeline in Docker"
	@echo "  make docker-model-pipeline   - Run model pipeline in Docker"
	@echo "  make docker-inference-pipeline - Run inference pipeline in Docker"
	@echo "  make docker-run-all      - Run all pipelines in Docker"
	@echo "  make docker-status       - Show Docker service status"
	@echo ""
	@echo "🔄 ML Pipeline Commands (Local):"
	@echo "  make data-pipeline       - Run the data preprocessing pipeline"
	@echo "  make train-pipeline      - Run the model training pipeline"
	@echo "  make inference-pipeline  - Run the batch inference pipeline"
	@echo ""
	@echo "📊 MLflow Commands:"
	@echo "  make mlflow-ui           - Launch MLflow UI (port $(MLFLOW_PORT))"
	@echo "  make stop-mlflow         - Stop MLflow servers"
	@echo ""
	@echo "🌐 S3 Commands:"
	@echo "  make s3-upload-data              - Upload data/raw & data/processed to S3 (one-time)"
	@echo "  make s3-list PREFIX=<prefix>     - List S3 keys with prefix"
	@echo "  make s3-clean                    - Clean project S3 artifacts (safe)"
	@echo "  make s3-delete-prefix PREFIX=<>  - Delete S3 keys with prefix (careful!)"
	@echo "  make s3-smoke                    - Test S3 connectivity"
	@echo ""
	@echo "🧪 Development Commands:"
	@echo "  make status              - Show project and S3 status"
	@echo "  make dev-install         - Install with development dependencies"
	@echo ""
	@echo "💡 Quick Start (Docker):"
	@echo "  1. Configure AWS credentials (~/.aws/credentials)"
	@echo "  2. cp .env.example .env && edit .env"
	@echo "  3. make docker-build"
	@echo "  4. make docker-up"
	@echo "  5. make docker-run-all"
	@echo ""
	@echo "💡 Quick Start (Local):"
	@echo "  1. make install"
	@echo "  2. Configure AWS settings in config.yaml (aws section)"
	@echo "  3. make s3-smoke"
	@echo "  4. make s3-upload-data (one-time setup)"
	@echo "  5. make data-pipeline"
	@echo "  6. make train-pipeline"
	@echo "  7. make inference-pipeline"

# ========================================================================================
# SETUP AND ENVIRONMENT COMMANDS
# ========================================================================================

# Install project dependencies and set up environment
install:
	@echo "📦 Installing project dependencies and setting up environment..."
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	@source .venv/bin/activate && pip install --upgrade pip
	@source .venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Installation completed successfully!"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Create necessary directories
setup-dirs:
	@echo "📁 Creating necessary directories..."
	@mkdir -p data/raw
	@echo "✅ Directories created successfully!"
	@echo "ℹ️  Note: Artifacts are now stored in S3, no local artifact directories needed"

# Clean up local cache only (safe cleanup)
clean:
	@echo "🧹 Cleaning up local cache and temporary files..."
	@rm -rf mlruns  # MLflow now uses S3 backend
	@find . -path "./.venv" -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -path "./.venv" -prune -o -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Local cleanup completed!"
	@echo "ℹ️  All artifacts are now stored in S3. Use 'make s3-clean' to clean S3 artifacts."

# Clean up S3 artifacts (project-specific only)
s3-clean:
	@echo "🗑️ Cleaning S3 ML pipeline artifacts..."
	@echo "⚠️ This will delete project artifacts from S3:"
	@echo "   • artifacts/data_artifacts/"
	@echo "   • artifacts/model_artifacts/"
	@echo "   • artifacts/inference_artifacts/"
	@echo "   • test/ (if exists)"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo ""
	@echo "🗑️ Deleting data artifacts..."
	@aws s3 rm s3://customer-churn-prediction-sahas-0200/artifacts/data_artifacts/ --recursive 2>/dev/null || echo "No data artifacts found"
	@echo "🗑️ Deleting model artifacts..."
	@aws s3 rm s3://customer-churn-prediction-sahas-0200/artifacts/model_artifacts/ --recursive 2>/dev/null || echo "No model artifacts found"
	@echo "🗑️ Deleting inference artifacts..."
	@aws s3 rm s3://customer-churn-prediction-sahas-0200/artifacts/inference_artifacts/ --recursive 2>/dev/null || echo "No inference artifacts found"
	@echo "🗑️ Deleting test artifacts..."
	@aws s3 rm s3://customer-churn-prediction-sahas-0200/test/ --recursive 2>/dev/null || echo "No test artifacts found"
	@echo "✅ S3 artifacts cleaned!"

# Remove legacy local artifact folders (since we're S3-only now)
clean-local-artifacts:
	@echo "🗑️  Removing legacy local artifact folders..."
	@echo "⚠️ This will permanently delete local folders:"
	@echo "   • artifacts/ (all local ML artifacts)"
	@echo "   • data/processed/ (processed data now in S3)"
	@echo "   • mlruns/ (MLflow now uses S3 backend)"
	@echo "   • test_*.py (test files)"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@rm -rf artifacts/
	@rm -rf data/processed/
	@rm -rf mlruns/
	@rm -f test_*.py
	@echo "✅ Legacy local artifacts removed - pipeline now uses S3-only!"

# ========================================================================================
# DOCKER COMMANDS
# ========================================================================================

# Build all Docker images
docker-build:
	@echo "🐳 Building all Docker images..."
	@docker-compose build --no-cache
	@echo "✅ All Docker images built successfully!"

# Start MLflow tracking server
docker-mlflow:
	@echo "🚀 Starting MLflow Tracking Server..."
	@docker-compose up -d mlflow-tracking
	@echo "✅ MLflow Tracking Server started at http://localhost:5001"

# Start all services
docker-up:
	@echo "🐳 Starting all Docker services..."
	@docker-compose up -d mlflow-tracking data-pipeline model-pipeline inference-pipeline
	@echo "✅ All Docker services started!"
	@echo "🌐 MLflow UI: http://localhost:5001"

# Stop all services
docker-down:
	@echo "🛑 Stopping all Docker services..."
	@docker-compose down
	@echo "✅ All Docker services stopped!"

# Clean up Docker resources
docker-clean:
	@echo "🧹 Cleaning up Docker resources..."
	@docker-compose down -v --remove-orphans
	@docker system prune -f
	@echo "✅ Docker cleanup completed!"

# Run data pipeline in Docker
docker-data-pipeline:
	@echo "🔄 Running data pipeline in Docker..."
	@docker-compose run --rm data-pipeline
	@echo "✅ Data pipeline completed!"

# Run model pipeline in Docker
docker-model-pipeline:
	@echo "🎯 Running model pipeline in Docker..."
	@docker-compose run --rm model-pipeline
	@echo "✅ Model pipeline completed!"

# Run inference pipeline in Docker
docker-inference-pipeline:
	@echo "🔮 Running inference pipeline in Docker..."
	@docker-compose run --rm inference-pipeline
	@echo "✅ Inference pipeline completed!"

# Run all pipelines in sequence (Docker)
docker-run-all: docker-mlflow
	@echo "🚀 Running all pipelines in Docker..."
	@sleep 10  # Wait for MLflow to be ready
	@docker-compose run --rm data-pipeline
	@docker-compose run --rm model-pipeline
	@docker-compose run --rm inference-pipeline
	@echo "🎉 All Docker pipelines completed!"

# Show Docker service status
docker-status:
	@echo "📊 Docker Service Status"
	@echo "======================="
	@docker-compose ps
	@echo ""
	@echo "🌐 MLflow UI: http://localhost:5001"

# View Docker logs
docker-logs:
	@echo "📋 Docker Service Logs"
	@echo "======================"
	@docker-compose logs --tail=50 -f

# ========================================================================================
# ML PIPELINE COMMANDS (Local)
# ========================================================================================

# Run data preprocessing pipeline
data-pipeline: setup-dirs
	@echo "🔄 Running data preprocessing pipeline..."
	@./run_local.sh python pipelines/data_pipeline.py
	@echo "✅ Data pipeline completed successfully!"

# Run model training pipeline
train-pipeline: setup-dirs
	@echo "🎯 Running model training pipeline..."
	@./run_local.sh python pipelines/training_pipeline.py
	@echo "✅ Training pipeline completed successfully!"

# Run batch inference pipeline
inference-pipeline: setup-dirs
	@echo "🔮 Running batch inference pipeline..."
	@./run_local.sh python pipelines/inference_pipeline.py
	@echo "✅ Inference pipeline completed successfully!"

# Run all pipelines in sequence
run-all: data-pipeline train-pipeline inference-pipeline
	@echo "🎉 All pipelines completed successfully!"

# ========================================================================================
# MLFLOW COMMANDS
# ========================================================================================

# Launch MLflow UI
mlflow-ui:
	@echo "📊 Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the server"
	@source $(VENV) && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)

# Stop MLflow servers
stop-mlflow:
	@echo "🛑 Stopping MLflow servers..."
	@-lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	@-ps aux | grep '[m]lflow ui' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@echo "✅ MLflow servers stopped"

# ========================================================================================
# S3 UTILITY COMMANDS
# ========================================================================================

# List S3 keys with prefix
s3-list:
	@echo "📋 Listing S3 keys..."
	@if [ -z "$(PREFIX)" ]; then echo "❌ Usage: make s3-list PREFIX=your-prefix"; exit 1; fi
	@aws s3 ls s3://customer-churn-prediction-sahas-0200/$(PREFIX) --recursive

# Delete specific S3 keys with prefix (use with caution)
s3-delete-prefix:
	@echo "🗑️ Deleting S3 keys with prefix: $(PREFIX)"
	@echo "⚠️ This will permanently delete data!"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@./run_local.sh python -c "from utils.s3_io import list_keys, delete_key; keys = list_keys('$(PREFIX)'); [delete_key(k) for k in keys]; print(f'Deleted {len(keys)} keys')"

# Upload local data folders to S3 (one-time setup)
s3-upload-data:
	@echo "📤 Uploading data folders to S3 (one-time setup)..."
	@echo "This will upload:"
	@echo "   • data/raw/ → s3://bucket/data/raw/"
	@echo "   • data/processed/ → s3://bucket/data/processed/"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	# Load .env into shell if present so environment variables (AWS_*) are exported for upload
	@set -o allexport; \
		if [ -f .env ]; then \
			sed -e 's/\r$$//' .env > .env.tmp && . .env.tmp && rm -f .env.tmp; \
		fi; \
		set +o allexport; \
	# Use run_local.sh to ensure venv activation and AWS credential helpers run
	@./run_local.sh python scripts/upload_data_to_s3.py

# S3 smoke test (write/read roundtrip)
s3-smoke:
	@echo "🔬 Running S3 smoke test..."
	@./run_local.sh python -c "from utils.s3_io import put_bytes, get_bytes; test_data = b'Hello S3!'; put_bytes(test_data, key='test/smoke_test.txt'); result = get_bytes('test/smoke_test.txt'); print('✅ S3 smoke test passed!' if result == test_data else '❌ S3 smoke test failed!')"

# Test MLflow S3 integration
mlflow-s3-test:
	@echo "🧪 Testing MLflow S3 integration..."
	@source $(VENV) && PYTHONPATH=. python test_mlflow_s3.py

# Show project status
status:
	@echo "📊 Project Status"
	@echo "================="
	@echo "Python Version: $$(python --version 2>/dev/null || echo 'Not found')"
	@echo "Virtual Env: $$([[ -d .venv ]] && echo '✅ Created' || echo '❌ Not found')"
	@echo "Dependencies: $$([[ -f .venv/pyvenv.cfg ]] && echo '✅ Installed' || echo '❌ Not installed')"
	@echo ""
	@echo "🌐 S3 Configuration:"
	@echo "S3 Bucket: $$(source $(VENV) && python -c 'from utils.config import get_s3_bucket; print(get_s3_bucket())' 2>/dev/null || echo 'Not configured')"
	@echo "AWS Region: $$(source $(VENV) && python -c 'from utils.config import get_aws_region; print(get_aws_region())' 2>/dev/null || echo 'Not configured')"
	@echo "Force S3 I/O: $$(source $(VENV) && python -c 'from utils.config import force_s3_io; print(force_s3_io())' 2>/dev/null || echo 'Not configured')"
	@echo ""
	@echo "📈 Recent S3 Artifacts:"
	@source $(VENV) && PYTHONPATH=. python -c "from utils.s3_io import list_keys; keys = list_keys('artifacts/'); print('\\n'.join(keys[:5]))" 2>/dev/null || echo "No S3 artifacts found"

# Install development dependencies
dev-install: install
	@echo "🛠️ Installing development dependencies..."
	@source $(VENV) && pip install -e ".[dev]"
	@echo "✅ Development setup completed!"