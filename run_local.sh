unset MLFLOW_TRACKING_URI
unset MLFLOW_DEFAULT_ARTIFACT_ROOT

# Set local MLflow tracking URI and Python path
export MLFLOW_TRACKING_URI="http://localhost:5001"
export PYTHONPATH="."
export AWS_PROFILE="${AWS_PROFILE:-default}"  # Use default AWS profile

# Set AWS credentials from environment or credentials file
if [ -z "$AWS_ACCESS_KEY_ID" ] && [ -f ~/.aws/credentials ]; then
    # Extract credentials from AWS credentials file
    AWS_ACCESS_KEY_ID=$(grep -A2 "\\[default\\]" ~/.aws/credentials | grep aws_access_key_id | cut -d'=' -f2 | tr -d ' ')
    AWS_SECRET_ACCESS_KEY=$(grep -A2 "\\[default\\]" ~/.aws/credentials | grep aws_secret_access_key | cut -d'=' -f2 | tr -d ' ')
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
    export AWS_REGION="${AWS_REGION:-ap-south-1}"
    
    # Set AWS config directory to avoid profile issues
    export AWS_CONFIG_FILE="$HOME/.aws/config"
    export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/credentials"
    
    echo "🔑 Loaded AWS credentials from ~/.aws/credentials"
elif [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "❌ ERROR: AWS credentials are required for local development!"
    echo "💡 Please set your AWS credentials:"
    echo "   Option 1 - Environment variables:"
    echo "     export AWS_ACCESS_KEY_ID='your-access-key'"
    echo "     export AWS_SECRET_ACCESS_KEY='your-secret-key'"
    echo "     export AWS_DEFAULT_REGION='ap-south-1'"
    echo ""
    echo "   Option 2 - AWS CLI configuration:"
    echo "     aws configure"
    echo ""
    exit 1
fi

echo "🧹 Cleaned Docker environment variables"
echo "🎯 Set MLFLOW_TRACKING_URI to: $MLFLOW_TRACKING_URI"
echo "🐍 Set PYTHONPATH to: $PYTHONPATH"
echo "🔑 AWS credentials: ${AWS_ACCESS_KEY_ID:0:5}***"

# Activate virtual environment
source .venv/bin/activate

# Run the command passed as argument
exec "$@"
