import json
import logging
import os
import joblib, sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
# Manual PySpark availability flag - set to False to prioritize pandas
PYSPARK_AVAILABLE = False  # Set to True to enable PySpark, False for pandas-only

# Conditional PySpark imports
if PYSPARK_AVAILABLE:
    try:
        from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
        from pyspark.sql import functions as F
        from utils.spark_session import get_or_create_spark_session
        from utils.spark_utils import spark_to_pandas
    except ImportError:
        PYSPARK_AVAILABLE = False
        SparkSession = None
        SparkDataFrame = None
        get_or_create_spark_session = None
        spark_to_pandas = None
else:
    SparkSession = None
    SparkDataFrame = None
    get_or_create_spark_session = None
    spark_to_pandas = None

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils.config import get_binning_config, get_encoding_config
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

""" 
    {
    "RowNumber": 1,
    "CustomerId": 15634602,
    "Firstname": "Grace",
    "Lastname": "Williams",
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
    }

"""
class ModelInference:
    """
    Enhanced model inference class with comprehensive logging and error handling.
    """
    
    def __init__(self, model_path: str, use_spark: bool = False, spark: Optional[SparkSession] = None):
        """
        Initialize the model inference system.
        
        Args:
            model_path: Path to the trained model file
            use_spark: Whether to use PySpark for preprocessing (default: False for single records)
            spark: Optional SparkSession instance
            
        Raises:
            ValueError: If model_path is invalid
            FileNotFoundError: If model file doesn't exist
        """
        logger.info(f"\n{'='*60}")
        logger.info("INITIALIZING MODEL INFERENCE")
        logger.info(f"{'='*60}")
        
        if not model_path or not isinstance(model_path, str):
            logger.error("✗ Invalid model path provided")
            raise ValueError("Invalid model path provided")
            
        self.model_path = model_path
        self.encoders = {}
        self.scaler = None
        self.scaler_metadata = None
        self.model = None
        self.linked_data_timestamp = None  # Will be set during model loading
        self.use_spark = use_spark
        self.spark = spark if spark else (get_or_create_spark_session() if use_spark else None)
        
        logger.info(f"Model Path: {model_path}")
        logger.info(f"Processing Engine: {'PySpark' if use_spark else 'Pandas'}")
        
        try:
            # Load model and configurations
            self.load_model()
            self.binning_config = get_binning_config()
            self.encoding_config = get_encoding_config()
            
            # Load scaler metadata
            self._load_scaler_metadata()
            
            logger.info("✓ Model inference system initialized successfully")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize model inference: {str(e)}")
            raise

    def load_model(self) -> None:
        """
        Load the trained model from disk with validation.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: For any loading errors
        """
        logger.info("Loading trained model...")
        
        # Quick local fallback first (development/ECS without S3 artifacts)
        try:
            artifacts_root = os.getenv('ARTIFACTS_ROOT', 'artifacts')
            local_model_path = os.path.join(artifacts_root, "models", "best_model.pkl")
            local_encoders_path = os.path.join(artifacts_root, "data", "encoders.pkl")
            local_scaler_path = os.path.join(artifacts_root, "data", "scaler.pkl")
            if os.path.exists(local_model_path):
                logger.info("🔄 Using LOCAL fallback model (best_model.pkl)")
                self.model = joblib.load(local_model_path)
                self.model_type = 'sklearn_local'
                # Load optional local encoders
                try:
                    if os.path.exists(local_encoders_path):
                        import pickle
                        with open(local_encoders_path, 'rb') as f:
                            self.encoders = pickle.load(f)
                        logger.info("✅ Local encoders loaded")
                except Exception as enc_err:
                    logger.warning(f"⚠️ Failed to load local encoders: {enc_err}")
                # Load optional local scaler
                try:
                    if os.path.exists(local_scaler_path):
                        import pickle
                        with open(local_scaler_path, 'rb') as f:
                            self.scaler = pickle.load(f)
                        logger.info("✅ Local scaler loaded")
                except Exception as sc_err:
                    logger.warning(f"⚠️ Failed to load local scaler: {sc_err}")
                return
        except Exception as local_quick_fallback_err:
            logger.warning(f"⚠️ Local quick fallback failed: {local_quick_fallback_err}")

        # Import S3 utilities for model loading (with unified timestamp resolver)
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
        from s3_io import read_pickle, key_exists
        from s3_artifact_manager import S3ArtifactManager
        from config import get_s3_bucket
        from timestamp_resolver import get_latest_train_timestamp
        
        bucket = get_s3_bucket()
        
        # Try to load model from local artifacts first, then S3
        logger.info("Loading model from local artifacts or S3...")
        
        import time
        import glob
        import json
        start_time = time.time()
        
        # First, try to find local model metadata files (using unified structure)
        local_model_metadata_pattern = "artifacts/train/*/model_metadata.json"
        local_metadata_files = glob.glob(local_model_metadata_pattern)
        
        if local_metadata_files:
            # Use the latest local metadata file
            latest_local_metadata = max(local_metadata_files, key=os.path.getctime)
            logger.info(f"Found local model metadata: {latest_local_metadata}")
            
            with open(latest_local_metadata, 'r') as f:
                metadata = json.load(f)
            
            train_artifacts_dir = os.path.dirname(latest_local_metadata)
            mlflow_model_path = metadata.get('mlflow_model_path')
            
            logger.info(f"📁 Using local model artifacts from: {train_artifacts_dir}")
            logger.info(f"📊 Model metadata: {metadata}")
            
        else:
            # Fallback to S3 if no local metadata found (using unified timestamp resolver)
            logger.info("No local model metadata found, trying S3 with unified timestamp resolver...")
            
            # Use unified timestamp resolver to get latest train timestamp
            latest_timestamp = get_latest_train_timestamp()
            if not latest_timestamp:
                logger.error(f"❌ No model timestamp folders found in S3")
                # Local simple fallback (development mode): use best_model.pkl and encoders/scaler from artifacts/
                try:
                    # In container, artifacts are copied to /opt/app/artifacts
                    artifacts_root = os.getenv('ARTIFACTS_ROOT', 'artifacts')
                    local_model_path = os.path.join(artifacts_root, "models", "best_model.pkl")
                    local_encoders_path = os.path.join(artifacts_root, "data", "encoders.pkl")
                    local_scaler_path = os.path.join(artifacts_root, "data", "scaler.pkl")
                    if os.path.exists(local_model_path):
                        logger.info("🔄 Falling back to local artifacts (development mode)")
                        logger.info(f"📦 Loading sklearn model from {local_model_path}")
                        self.model = joblib.load(local_model_path)
                        self.model_type = 'sklearn_local'
                        # Load local encoders if present
                        if os.path.exists(local_encoders_path):
                            try:
                                import pickle
                                with open(local_encoders_path, 'rb') as f:
                                    self.encoders = pickle.load(f)
                                logger.info("✅ Local encoders loaded")
                            except Exception as enc_err:
                                logger.warning(f"⚠️ Failed to load local encoders: {enc_err}")
                        # Load local scaler if present
                        if os.path.exists(local_scaler_path):
                            try:
                                import pickle
                                with open(local_scaler_path, 'rb') as f:
                                    self.scaler = pickle.load(f)
                                logger.info("✅ Local scaler loaded")
                            except Exception as sc_err:
                                logger.warning(f"⚠️ Failed to load local scaler: {sc_err}")
                        return
                except Exception as local_fallback_err:
                    logger.warning(f"⚠️ Local fallback failed: {local_fallback_err}")
                raise FileNotFoundError(f"No model artifacts found in S3")
                
            # Use unified S3 structure: artifacts/train/<TIMESTAMP>/
            train_artifacts_dir = f"artifacts/train/{latest_timestamp}"
            metadata_s3_key = f"{train_artifacts_dir}/model_metadata.json"
            
            logger.info(f"📁 Using latest train artifacts (unified structure): {train_artifacts_dir}")
            logger.info(f"🔍 Looking for metadata: s3://{bucket}/{metadata_s3_key}")
            
            # Try to read the metadata (optional - we can proceed without it)
            metadata = {}
            model_name = None
            mlflow_model_path = None
            
            try:
                from utils.s3_io import get_bytes, key_exists
                if key_exists(metadata_s3_key):
                    metadata_bytes = get_bytes(metadata_s3_key)
                    metadata = json.loads(metadata_bytes.decode('utf-8'))
                    model_name = metadata.get('model_name')
                    mlflow_model_path = metadata.get('mlflow_model_path')
                    
                    # Store the linked data artifacts timestamp for encoder/scaler loading
                    self.linked_data_timestamp = metadata.get('data_artifacts_timestamp', latest_timestamp)
                    
                    logger.info(f"✅ Model metadata loaded:")
                    logger.info(f"  • Model name: {model_name}")
                    logger.info(f"  • MLflow path: {mlflow_model_path}")
                    logger.info(f"  • Linked data timestamp: {self.linked_data_timestamp}")
                else:
                    logger.info(f"⚠️ No metadata file found, proceeding with dynamic discovery...")
                    # Fallback: use the same timestamp as the model
                    self.linked_data_timestamp = latest_timestamp
            except Exception as metadata_error:
                logger.warning(f"⚠️ Failed to load metadata (proceeding anyway): {metadata_error}")
                # Fallback: use the same timestamp as the model
                self.linked_data_timestamp = latest_timestamp
                
        # Try to load the Spark model directly from local or S3 model artifacts directory
        try:
            spark_available = False
            try:
                from pyspark.ml import PipelineModel  # type: ignore
                spark_available = True
            except Exception as e:
                logger.info(f"PySpark not available ({e}); proceeding with sklearn fallback")

            if spark_available and local_metadata_files:
                # Load from local model artifacts (unified structure)
                spark_model_path = metadata.get('spark_model_path', f"artifacts/train/{metadata['timestamp']}/spark_model")
                logger.info(f"🔍 Attempting to load Spark model from local path: {spark_model_path}")
                
                if os.path.exists(spark_model_path):
                    self.model = PipelineModel.load(spark_model_path)
                    self.model_type = 'spark_s3'
                    logger.info(f"✅ Successfully loaded local Spark model from: {spark_model_path}")
                    
                    # Load encoders from local files
                    if 'encoders' in metadata:
                        for encoder_name, encoder_path in metadata['encoders'].items():
                            if os.path.exists(encoder_path):
                                logger.info(f"📋 Loading local encoder: {encoder_name} from {encoder_path}")
                                with open(encoder_path, 'r') as f:
                                    encoder_data = json.load(f)
                                    self.encoders[encoder_name] = encoder_data
                            else:
                                logger.warning(f"⚠️ Local encoder not found: {encoder_path}")
                    
                    load_time = time.time() - start_time
                    logger.info(f"✅ Local model loaded successfully in {load_time:.2f} seconds")
                    return
                else:
                    raise FileNotFoundError(f"Local Spark model not found at: {spark_model_path}")
            if spark_available:
                # Construct S3A path for the Spark model directory
                # The model should be saved as a directory in S3 model artifacts
                spark_model_s3a_path = f"s3a://{bucket}/{train_artifacts_dir}/spark_model"
                
                logger.info(f"🔍 Attempting to load Spark model from: {spark_model_s3a_path}")
                
                # Try to load the PipelineModel directly from S3A
                try:
                    self.model = PipelineModel.load(spark_model_s3a_path)
                    self.model_type = 'spark_s3'
                    
                    load_time = time.time() - start_time
                    logger.info(f"✅ Model loaded directly from S3 in {load_time:.2f} seconds")
                    logger.info(f"  • Model path: {spark_model_s3a_path}")
                    logger.info(f"  • Model stages: {len(self.model.stages)}")
                    
                    # Load encoders from S3 data artifacts folder (unified structure)
                    self._load_encoders_from_s3()
                    return
                    
                except Exception as s3_load_error:
                    logger.warning(f"⚠️ Direct S3 Spark model loading failed: {s3_load_error}")
                    logger.info("🔄 Trying sklearn model fallback...")
            
            # If no Spark available (or Spark loading failed), attempt sklearn directly
            from utils.s3_io import read_pickle, key_exists
            sklearn_model_key = f"{train_artifacts_dir}/churn_model.pkl"
            if key_exists(sklearn_model_key):
                logger.info(f"🔍 Attempting to load sklearn model from: s3://{bucket}/{sklearn_model_key}")
                self.model = read_pickle(key=sklearn_model_key, use_joblib=True)
                self.model_type = 'sklearn_s3'
                load_time = time.time() - start_time
                logger.info(f"✅ Sklearn model loaded from S3 in {load_time:.2f} seconds")
                logger.info(f"  • Model type: {type(self.model).__name__}")
                self._load_encoders_from_s3()
                return
            
            # If MLflow path points to a Spark model and Spark isn't available, skip MLflow spark load
            logger.warning("No Spark model available and sklearn model not found in S3.")
            raise FileNotFoundError("No usable model artifacts found (Spark unavailable and sklearn model missing)")
                    
        except FileNotFoundError as not_found:
            # Re-raise with the same message
            raise not_found
        except Exception as metadata_error:
            logger.error(f"❌ Model loading error: {metadata_error}")
            # As a final fallback, try local sklearn again before failing
            try:
                artifacts_root = os.getenv('ARTIFACTS_ROOT', 'artifacts')
                local_model_path = os.path.join(artifacts_root, "models", "best_model.pkl")
                if os.path.exists(local_model_path):
                    logger.info("🔄 Final fallback: using local sklearn model best_model.pkl")
                    self.model = joblib.load(local_model_path)
                    self.model_type = 'sklearn_local'
                    return
            except Exception:
                pass
            raise FileNotFoundError(f"Model loading failed: {metadata_error}")
            
        logger.error(f"✗ No model metadata found for base name: {self.model_path}")
        raise FileNotFoundError(f"No model metadata found with base name: {self.model_path}")

    def _get_latest_model_timestamp_from_s3(self, bucket: str) -> Optional[str]:
        """
        DEPRECATED: Use unified timestamp resolver instead.
        Dynamically get the latest model timestamp folder from S3.
        
        Args:
            bucket: S3 bucket name
            
        Returns:
            Latest timestamp string or None if no folders found
        """
        logger.warning("⚠️ Using deprecated _get_latest_model_timestamp_from_s3. Use timestamp_resolver.get_latest_train_timestamp() instead.")
        try:
            from utils.timestamp_resolver import get_latest_train_timestamp
            return get_latest_train_timestamp()
        except Exception as e:
            logger.error(f"❌ Failed to get latest timestamp: {e}")
            # Fallback to old method for backward compatibility
            try:
                from utils.s3_io import list_keys
                
                # List all keys in unified train artifacts directory
                prefix = "artifacts/train/"
                keys = list_keys(prefix=prefix)
                
                # Extract timestamp folders (format: YYYYMMDDHHMMSS)
                timestamps = set()
                for key in keys:
                    # Remove prefix and get the first path component (timestamp)
                    relative_path = key[len(prefix):]
                    if '/' in relative_path:
                        timestamp_candidate = relative_path.split('/')[0]
                        # Validate it's a timestamp (14 digits)
                        if timestamp_candidate.isdigit() and len(timestamp_candidate) == 14:
                            timestamps.add(timestamp_candidate)
                
                if not timestamps:
                    logger.warning(f"⚠️ No timestamp folders found in s3://{bucket}/{prefix}")
                    return None
                    
                # Get the latest (max) timestamp
                latest_timestamp = max(timestamps)
                logger.info(f"📅 Found {len(timestamps)} timestamp folders, using latest: {latest_timestamp}")
                return latest_timestamp
                
            except Exception as fallback_e:
                logger.error(f"❌ Fallback also failed: {fallback_e}")
                return None
    
    def _get_latest_data_timestamp_from_s3(self, bucket: str) -> Optional[str]:
        """
        DEPRECATED: Use unified timestamp resolver instead.
        Dynamically get the latest data artifacts timestamp folder from S3.
        
        Args:
            bucket: S3 bucket name
            
        Returns:
            Latest timestamp string or None if no folders found
        """
        logger.warning("⚠️ Using deprecated _get_latest_data_timestamp_from_s3. Use timestamp_resolver.get_latest_data_timestamp() instead.")
        try:
            from utils.timestamp_resolver import get_latest_data_timestamp
            return get_latest_data_timestamp()
        except Exception as e:
            logger.error(f"❌ Failed to get latest data timestamp: {e}")
            # Fallback to old method for backward compatibility
            try:
                from utils.s3_io import list_keys
                
                # List all keys in unified data artifacts directory
                prefix = "artifacts/data/"
                keys = list_keys(prefix=prefix)
                
                # Extract timestamp folders (format: YYYYMMDDHHMMSS)
                timestamps = set()
                for key in keys:
                    # Remove prefix and get the first path component (timestamp)
                    relative_path = key[len(prefix):]
                    if '/' in relative_path:
                        timestamp_candidate = relative_path.split('/')[0]
                        # Validate it's a timestamp (14 digits)
                        if timestamp_candidate.isdigit() and len(timestamp_candidate) == 14:
                            timestamps.add(timestamp_candidate)
                
                if not timestamps:
                    logger.warning(f"⚠️ No data timestamp folders found in s3://{bucket}/{prefix}")
                    return None
                    
                # Get the latest (max) timestamp
                latest_timestamp = max(timestamps)
                logger.info(f"📅 Found {len(timestamps)} data timestamp folders, using latest: {latest_timestamp}")
                return latest_timestamp
                
            except Exception as fallback_e:
                logger.error(f"❌ Fallback also failed: {fallback_e}")
                return None
    
    def _load_encoders_from_s3(self) -> None:
        """Load feature encoders from S3 using dynamic timestamp discovery"""
        try:
            import json
            from utils.s3_io import get_bytes, key_exists
            from utils.config import get_s3_bucket
            
            bucket = get_s3_bucket()
            logger.info("Loading feature encoders from S3...")
            
            # Use the linked data timestamp from model metadata (if available)
            if self.linked_data_timestamp:
                data_timestamp = self.linked_data_timestamp
                logger.info(f"📅 Using linked data artifacts timestamp from model metadata: {data_timestamp}")
            else:
                # Fallback: Get latest data artifacts timestamp dynamically
                data_timestamp = self._get_latest_data_timestamp_from_s3(bucket)
                if not data_timestamp:
                    logger.warning("⚠️ No data artifacts timestamp folders found in S3")
                    logger.info("Continuing without encoders - some preprocessing steps may be skipped")
                    return
                logger.info(f"📅 Using latest data artifacts timestamp (fallback): {data_timestamp}")
            
            # List of encoder files to look for (lowercase filenames)
            encoder_files = ['gender_encoder.json', 'geography_encoder.json']
            
            for encoder_file in encoder_files:
                encoder_name = encoder_file.replace('_encoder.json', '').capitalize()
                encoder_path = f"artifacts/data/{data_timestamp}/{encoder_file}"
                
                try:
                    if key_exists(encoder_path):
                        logger.info(f"🔍 Loading {encoder_name} encoder from: s3://{bucket}/{encoder_path}")
                        encoder_bytes = get_bytes(encoder_path)
                        encoder_data = json.loads(encoder_bytes.decode('utf-8'))
                        self.encoders[encoder_name] = encoder_data
                        logger.info(f"✅ {encoder_name} encoder loaded: type={encoder_data.get('encoder_type', 'unknown')}, categories={encoder_data.get('categories', [])}")
                    else:
                        logger.warning(f"⚠️ Encoder not found: {encoder_path}")
                        
                except Exception as encoder_error:
                    logger.warning(f"⚠️ Failed to load {encoder_name} encoder: {encoder_error}")
            
            logger.info(f"✓ Loaded {len(self.encoders)} feature encoders from S3")
            
        except Exception as e:
            logger.warning(f"⚠ Failed to load encoders from S3: {e}")
            logger.info("Continuing without encoders - some preprocessing steps may be skipped")
    
    def _load_scaler_metadata(self) -> None:
        """Load scaler metadata from S3 for inference"""
        try:
            import json
            from utils.s3_io import get_bytes, key_exists, read_pickle
            from utils.config import get_s3_bucket
            
            bucket = get_s3_bucket()
            logger.info("Loading scaler metadata from S3...")
            
            # Use the linked data timestamp from model metadata (if available)
            if self.linked_data_timestamp:
                data_timestamp = self.linked_data_timestamp
                logger.info(f"📅 Using linked data artifacts timestamp from model metadata: {data_timestamp}")
            else:
                # Fallback: Get latest data artifacts timestamp dynamically
                data_timestamp = self._get_latest_data_timestamp_from_s3(bucket)
                if not data_timestamp:
                    logger.warning("⚠️ No scaler metadata found - scaling will be skipped")
                    return
                logger.info(f"📅 Using latest data artifacts timestamp (fallback): {data_timestamp}")
            
            # Load scaler object
            scaler_path = f"artifacts/data/{data_timestamp}/scaler.pkl"
            try:
                if key_exists(scaler_path):
                    self.scaler = read_pickle(key=scaler_path, use_joblib=True)
                    logger.info(f"✅ Scaler object loaded from: s3://{bucket}/{scaler_path}")
                else:
                    logger.warning(f"⚠️ Scaler not found: {scaler_path}")
                    return
            except Exception as scaler_error:
                logger.warning(f"⚠️ Failed to load scaler: {scaler_error}")
                return
            
            # Load scaler metadata
            metadata_path = f"artifacts/data/{data_timestamp}/scaler_metadata.json"
            try:
                if key_exists(metadata_path):
                    metadata_bytes = get_bytes(metadata_path)
                    self.scaler_metadata = json.loads(metadata_bytes.decode('utf-8'))
                    logger.info(f"✅ Scaler metadata loaded from: s3://{bucket}/{metadata_path}")
                    logger.info(f"  • Columns to scale: {self.scaler_metadata.get('columns_to_scale', [])}")
                    logger.info(f"  • Scaling type: {self.scaler_metadata.get('scaling_type', 'unknown')}")
                else:
                    logger.warning(f"⚠️ Scaler metadata not found: {metadata_path}")
            except Exception as metadata_error:
                logger.warning(f"⚠️ Failed to load scaler metadata: {metadata_error}")
                
        except Exception as e:
            logger.warning(f"⚠ Failed to load scaler: {e}")
            logger.info("Continuing without scaler - features will not be scaled")

    def load_encoders(self, encoders_dir: str) -> None:
        """
        Load feature encoders from directory with validation and logging.
        
        Args:
            encoders_dir: Directory containing encoder JSON files
            
        Raises:
            FileNotFoundError: If encoders directory doesn't exist
            Exception: For any loading errors
        """
        logger.info(f"\n{'='*50}")
        logger.info("LOADING FEATURE ENCODERS")
        logger.info(f"{'='*50}")
        
        if not os.path.exists(encoders_dir):
            logger.error(f"✗ Encoders directory not found: {encoders_dir}")
            raise FileNotFoundError(f"Encoders directory not found: {encoders_dir}")
        
        try:
            encoder_files = [f for f in os.listdir(encoders_dir) if f.endswith('_encoder.json')]
            
            if not encoder_files:
                logger.warning("⚠ No encoder files found in directory")
                return
            
            logger.info(f"Found {len(encoder_files)} encoder files")
            
            for file in encoder_files:
                feature_name = file.split('_encoder.json')[0]
                file_path = os.path.join(encoders_dir, file)
                
                with open(file_path, 'r') as f:
                    encoder_data = json.load(f)
                    self.encoders[feature_name] = encoder_data
                    
                logger.info(f"  ✓ Loaded encoder for '{feature_name}': {len(encoder_data)} mappings")
            
            logger.info(f"✓ All encoders loaded successfully")
            logger.info(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to load encoders: {str(e)}")
            raise

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for model prediction with ONE-HOT encoding and proper scaling.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Preprocessed DataFrame ready for prediction
            
        Raises:
            ValueError: If input data is invalid
            Exception: For any preprocessing errors
        """
        logger.info(f"\n{'='*50}")
        logger.info("PREPROCESSING INPUT DATA")
        logger.info(f"{'='*50}")
        
        # Check if data is valid (works for both DataFrame and dict)
        if isinstance(data, pd.DataFrame):
            if data.empty:
                logger.error("✗ Input DataFrame cannot be empty")
                raise ValueError("Input DataFrame cannot be empty")
            df = data.copy()
            logger.info(f"✓ Input data received as DataFrame: {df.shape}")
        elif isinstance(data, dict):
            if not data:
                logger.error("✗ Input data must be a non-empty dictionary")
                raise ValueError("Input data must be a non-empty dictionary")
            # Convert to DataFrame
            df = pd.DataFrame([data])
            logger.info(f"✓ Input data converted to DataFrame: {df.shape}")
        else:
            logger.error(f"✗ Input data must be a DataFrame or dict, got: {type(data)}")
            raise ValueError(f"Input data must be a DataFrame or dict, got: {type(data)}")
        
        try:
            logger.info(f"  • Input features: {list(df.columns)}")
            
            # Drop unnecessary columns first
            drop_columns = ['RowNumber', 'CustomerId', 'Firstname', 'Lastname']
            existing_drop_columns = [col for col in drop_columns if col in df.columns]
            if existing_drop_columns:
                df = df.drop(columns=existing_drop_columns)
                logger.info(f"  ✓ Dropped columns: {existing_drop_columns}")
            
            # Apply ONE-HOT encoding for categorical variables
            if self.encoders:
                logger.info("Applying ONE-HOT encoding for categorical features...")
                for col, encoder_info in self.encoders.items():
                    if col in df.columns:
                        original_value = df[col].iloc[0]
                        encoder_type = encoder_info.get('encoder_type', 'unknown')
                        
                        if encoder_type == 'one_hot':
                            # Get categories from encoder
                            categories = encoder_info.get('categories', [])
                            logger.info(f"  • {col}: {original_value} → one-hot encoding")
                            
                            # Create binary columns for each category
                            for category in categories:
                                new_col_name = f"{col}_{category}"
                                df[new_col_name] = (df[col] == category).astype(int)
                            
                            # Drop original column
                            df = df.drop(columns=[col])
                            
                            binary_cols = [f"{col}_{cat}" for cat in categories]
                            logger.info(f"    ✓ Created binary columns: {binary_cols}")
                        else:
                            logger.warning(f"  ⚠ Unknown encoder type '{encoder_type}' for {col}")
                    else:
                        logger.warning(f"  ⚠ Column '{col}' not found in input data")
            else:
                logger.warning("No encoders available - skipping encoding step")

            # Apply feature binning (KEEP CreditScore)
            if 'CreditScore' in df.columns:
                logger.info("Applying feature binning for CreditScore...")
                original_score = df['CreditScore'].iloc[0]
                
                def bin_credit_score(score):
                    if score is None or pd.isna(score):
                        return 2  # Default to 'Good'
                    elif score <= 580:
                        return 0  # Poor
                    elif score <= 670:
                        return 1  # Fair
                    elif score <= 740:
                        return 2  # Good
                    elif score <= 800:
                        return 3  # Very Good
                    else:
                        return 4  # Excellent
                
                df['CreditScoreBins'] = df['CreditScore'].apply(bin_credit_score)
                # CRITICAL: Do NOT drop CreditScore - model expects both columns
                
                binned_score = df['CreditScoreBins'].iloc[0]
                bin_names = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
                logger.info(f"  ✓ CreditScore: {original_score} → CreditScoreBins: {binned_score} ({bin_names[binned_score] if binned_score < len(bin_names) else 'Unknown'})")
                logger.info(f"  ✓ Kept 'CreditScore' column for model compatibility")
            else:
                logger.warning("  ⚠ CreditScore not found - skipping binning")
            
            # Apply feature scaling using loaded scaler
            if self.scaler is not None and self.scaler_metadata is not None:
                columns_to_scale = self.scaler_metadata.get('columns_to_scale', [])
                available_scale_cols = [col for col in columns_to_scale if col in df.columns]
                
                if available_scale_cols:
                    logger.info(f"Applying feature scaling to {len(available_scale_cols)} columns: {available_scale_cols}")
                    
                    # Log before scaling
                    for col in available_scale_cols:
                        logger.info(f"  • {col}: {df[col].iloc[0]:.4f} (before scaling)")
                    
                    # Apply scaling using pre-fitted scaler (NO re-fitting)
                    df[available_scale_cols] = self.scaler.transform(df[available_scale_cols])
                    
                    # Log after scaling
                    for col in available_scale_cols:
                        logger.info(f"  ✓ {col}: {df[col].iloc[0]:.4f} (after scaling)")
                else:
                    logger.warning(f"  ⚠ No scalable columns found in data")
            else:
                logger.warning("No scaler available - skipping scaling step")
            
            # Ensure expected one-hot columns exist even if encoders are missing
            expected_one_hot_cols = [
                'Geography_France', 'Geography_Germany', 'Geography_Spain',
                'Gender_Female', 'Gender_Male'
            ]
            for col in expected_one_hot_cols:
                if col not in df.columns:
                    df[col] = 0

            # CRITICAL: Enforce exact column order for model compatibility
            logger.info("\n🔧 Enforcing exact column order for model compatibility...")
            expected_column_order = [
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'CreditScoreBins',
                'Geography_France', 'Geography_Germany', 'Geography_Spain',
                'Gender_Female', 'Gender_Male'
            ]
            
            # Filter to only columns that exist
            available_columns = [col for col in expected_column_order if col in df.columns]
            missing_columns = [col for col in expected_column_order if col not in df.columns]
            extra_columns = [col for col in df.columns if col not in expected_column_order]
            
            if missing_columns:
                logger.warning(f"  ⚠️ Missing expected columns: {missing_columns}")
            if extra_columns:
                logger.warning(f"  ⚠️ Extra columns (will be dropped): {extra_columns}")
            
            # Reorder columns
            df = df[available_columns]
            
            logger.info(f"  ✅ Columns reordered: {available_columns}")
            logger.info(f"  • Total columns: {len(available_columns)}")
            
            logger.info(f"✓ Preprocessing completed - Final shape: {df.shape}")
            logger.info(f"{'='*50}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {str(e)}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            raise
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Make prediction on input data with comprehensive logging.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary containing prediction status and confidence
            
        Raises:
            ValueError: If input data is invalid
            Exception: For any prediction errors
        """
        logger.info(f"\n{'='*60}")
        logger.info("MAKING PREDICTION")
        logger.info(f"{'='*60}")
        
        # Check if data is empty (works for both DataFrame and dict)
        if isinstance(data, pd.DataFrame):
            if data.empty:
                logger.error("✗ Input data cannot be empty")
                raise ValueError("Input data cannot be empty")
        elif isinstance(data, dict):
            if not data:
                logger.error("✗ Input data cannot be empty")
                raise ValueError("Input data cannot be empty")
        elif data is None:
            logger.error("✗ Input data cannot be None")
            raise ValueError("Input data cannot be None")
        
        if self.model is None:
            logger.error("✗ Model not loaded")
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess input data
            processed_data = self.preprocess_input(data)
            
            # Make prediction based on model type
            logger.info("Generating predictions...")
            
            if hasattr(self, 'model_type') and self.model_type in ['pyspark', 'spark_mlflow', 'spark_s3']:
                # PySpark model prediction
                spark_df = self.spark.createDataFrame(processed_data)
                predictions = self.model.transform(spark_df)
                
                # Get prediction and probability
                prediction_row = predictions.select("prediction", "probability").collect()[0]
                prediction = int(prediction_row.prediction)
                
                # Extract probability for positive class (index 1)
                probability_vector = prediction_row.probability
                probability = float(probability_vector[1])
                
            elif hasattr(self, 'model_type') and self.model_type == 'sklearn_s3':
                # Scikit-learn model prediction (from S3 fallback)
                logger.info("Using sklearn model for prediction...")
                y_pred = self.model.predict(processed_data)
                y_proba = self.model.predict_proba(processed_data)[:, 1]
                
                prediction = int(y_pred[0])
                probability = float(y_proba[0])
                
            else:
                # Default scikit-learn model prediction
                y_pred = self.model.predict(processed_data)
                y_proba = self.model.predict_proba(processed_data)[:, 1]
                
                prediction = int(y_pred[0])
                probability = float(y_proba[0])
            
            status = 'Churn' if prediction == 1 else 'Retain'
            confidence = round(probability * 100, 2)
            
            result = {
                "Status": status,
                "Confidence": f"{confidence}%"
            }
            
            logger.info("✓ Prediction completed:")
            logger.info(f"  • Raw Prediction: {prediction}")
            logger.info(f"  • Raw Probability: {probability:.4f}")
            logger.info(f"  • Final Status: {status}")
            logger.info(f"  • Confidence: {confidence}%")
            logger.info(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Prediction failed: {str(e)}")
            raise