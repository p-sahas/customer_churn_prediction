"""
PySpark-based data processing pipeline for customer churn prediction.
Supports both CSV and Parquet output formats with comprehensive preprocessing.
"""

import os
import sys
import logging
import json
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Ensure project root and `src` are on sys.path so top-level packages like `utils` and modules in `src` are importable
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.spark_session import create_spark_session, stop_spark_session
from utils.spark_utils import save_dataframe, spark_to_pandas, get_dataframe_info, check_missing_values
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from data_splitter import SimpleTrainTestSplitStrategy

from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config, get_s3_bucket, force_s3_io
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
from s3_artifact_manager import S3ArtifactManager, get_s3_artifact_paths
from s3_io import write_df_csv, write_pickle
import mlflow



def log_stage_metrics(df: DataFrame, stage: str, additional_metrics: Dict = None, spark: SparkSession = None):
    """Log key metrics for each processing stage."""
    try:
        # Calculate missing values count efficiently
        missing_counts = []
        for col in df.columns:
            missing_counts.append(df.filter(F.col(col).isNull()).count())
        total_missing = sum(missing_counts)
        
        metrics = {
            f'{stage}_rows': df.count(),
            f'{stage}_columns': len(df.columns),
            f'{stage}_missing_values': total_missing,
            f'{stage}_partitions': df.rdd.getNumPartitions()
        }
        
        if additional_metrics:
            metrics.update({f'{stage}_{k}': v for k, v in additional_metrics.items()})
        
        mlflow.log_metrics(metrics)
        logger.info(f"✓ Metrics logged for {stage}: ({metrics[f'{stage}_rows']}, {metrics[f'{stage}_columns']})")
        
    except Exception as e:
        logger.error(f"✗ Failed to log metrics for {stage}: {str(e)}")


def save_processed_data(
    X_train: DataFrame, 
    X_test: DataFrame, 
    Y_train: DataFrame, 
    Y_test: DataFrame,
    pipeline_timestamp: str,
    output_format: str = "both"
) -> Dict[str, str]:
    """
    Save processed data to S3 in specified format(s) with timestamp-based naming.
    
    Args:
        X_train, X_test, Y_train, Y_test: PySpark DataFrames
        pipeline_timestamp: Timestamp for this pipeline run
        output_format: "csv", "parquet", or "both"
        
    Returns:
        Dictionary of S3 key paths with timestamp
    """
    paths = {}
    s3_manager = S3ArtifactManager()
    bucket = get_s3_bucket()
    
    logger.info(f"💾 Saving artifacts to S3 with timestamp: {pipeline_timestamp}")
    
    if output_format in ["csv", "both"]:
        # Save as CSV to S3
        logger.info("Saving data in CSV format to S3...")
        
        # Convert to pandas for CSV upload
        X_train_pd = spark_to_pandas(X_train)
        X_test_pd = spark_to_pandas(X_test)
        Y_train_pd = spark_to_pandas(Y_train)
        Y_test_pd = spark_to_pandas(Y_test)
        
        # Create S3 CSV paths for data artifacts
        csv_paths = s3_manager.create_s3_paths(
            ['X_train', 'X_test', 'Y_train', 'Y_test'], 
            timestamp=pipeline_timestamp,
            artifact_type='data_artifacts',
            format_ext='csv'
        )
        
        # Try to upload CSV files to S3, fallback to local if needed
        try:
            write_df_csv(X_train_pd, key=csv_paths['X_train'])
            write_df_csv(X_test_pd, key=csv_paths['X_test'])
            write_df_csv(Y_train_pd, key=csv_paths['Y_train'])
            write_df_csv(Y_test_pd, key=csv_paths['Y_test'])
            
            paths.update({f"{k}_csv": v for k, v in csv_paths.items()})
            logger.info("✓ CSV files saved to S3")
            
        except Exception as s3_error:
            logger.error(f"❌ S3 save failed: {s3_error}")
            logger.error("💡 Please check your AWS credentials and S3 bucket configuration")
            raise s3_error
    
    # Clean up old artifacts in S3 (keep last 5 versions) - optional
    try:
        s3_manager.cleanup_old_artifacts(artifact_type='data_artifacts', keep_count=5)
    except Exception as cleanup_error:
        logger.warning(f"S3 cleanup failed: {cleanup_error}")
    
    paths['timestamp'] = pipeline_timestamp
    return paths


def data_pipeline(
    data_path: str = None,
    target_column: str = 'Exited',
    test_size: float = 0.2,
    force_rebuild: bool = False,
    output_format: str = "both"
) -> Dict[str, np.ndarray]:
    """
    Execute comprehensive data processing pipeline with PySpark and MLflow tracking.
    
    Args:
        data_path: Path to the raw data file
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        force_rebuild: Whether to force rebuild of existing artifacts
        output_format: Output format - "csv", "parquet", or "both"
        
    Returns:
        Dictionary containing processed train/test splits as numpy arrays
    """
    
    # Get data path from config if not provided
    if data_path is None:
        from utils.config import load_config
        config = load_config()
        data_path = config['data_paths']['raw_data']
        logger.info(f"📁 Using data path from config: {data_path}")
    
    # Generate single timestamp for entire pipeline run
    from datetime import datetime
    pipeline_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    logger.info(f"🕐 Pipeline timestamp: {pipeline_timestamp}")
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING PYSPARK DATA PIPELINE")
    logger.info(f"{'='*80}")
    
    # Input validation for local files (skip for S3 paths)
    if not data_path.startswith('s3://') and not os.path.exists(data_path):
        logger.error(f"✗ Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not 0 < test_size < 1:
        logger.error(f"✗ Invalid test_size: {test_size}")
        raise ValueError(f"Invalid test_size: {test_size}")
    
    # Initialize Spark session
    spark = create_spark_session("ChurnPredictionDataPipeline")
    
    try:
        # Load configurations
        data_paths = get_data_paths()
        columns = get_columns()
        outlier_config = get_outlier_config()
        binning_config = get_binning_config()
        encoding_config = get_encoding_config()
        scaling_config = get_scaling_config()
        splitting_config = get_splitting_config()
        
        # Initialize MLflow tracking
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags('data_pipeline_pyspark', {
            'data_source': data_path,
            'force_rebuild': str(force_rebuild),
            'target_column': target_column,
            'output_format': output_format,
            'processing_engine': 'pyspark'
        })
        run = mlflow_tracker.start_run(run_name='data_pipeline_pyspark', tags=run_tags)
        
        # MLflow artifacts are now handled by S3 backend, no local directory needed
        
        # Check for existing artifacts in S3
        s3_manager = S3ArtifactManager()
        try:
            latest_paths = s3_manager.get_latest_artifacts(['X_train', 'X_test', 'Y_train', 'Y_test'], artifact_type='data_artifacts', format_ext='csv')
            artifacts_exist = len(latest_paths) == 4
        except Exception as e:
            logger.info(f"Could not check S3 artifacts: {e}")
            artifacts_exist = False
        
        if artifacts_exist and not force_rebuild:
            logger.info("✓ Loading existing processed data artifacts from S3")
            from s3_io import read_df_csv
            X_train = read_df_csv(key=latest_paths['X_train'])
            X_test = read_df_csv(key=latest_paths['X_test'])
            Y_train = read_df_csv(key=latest_paths['Y_train'])
            Y_test = read_df_csv(key=latest_paths['Y_test'])
            
            mlflow_tracker.log_data_pipeline_metrics({
                'total_samples': len(X_train) + len(X_test),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'processing_engine': 'existing_artifacts'
            })
            mlflow_tracker.end_run()
            
            logger.info("✓ Data pipeline completed using existing artifacts")
            return {
                'X_train': X_train.values,
                'X_test': X_test.values,
                'Y_train': Y_train.values.ravel(),
                'Y_test': Y_test.values.ravel()
            }
        
        # Process data from scratch with PySpark
        logger.info("Processing data from scratch with PySpark...")
        
        # Data ingestion
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA INGESTION STEP")
        logger.info(f"{'='*80}")
        
        # Check if we should load from S3 or local
        if force_s3_io():
            # Try to load from S3 first using boto3 (more reliable than S3A)
            from s3_io import key_exists, read_df_csv
            s3_key = data_path  # Use local path as S3 key (data/raw/ChurnModelling.csv)
            bucket = get_s3_bucket()
            
            if key_exists(s3_key):
                logger.info(f"📁 Loading raw data from S3 using boto3: s3://{bucket}/{s3_key}")
                # Use boto3 to load CSV instead of S3A (more reliable)
                df_pandas = read_df_csv(key=s3_key)
                # Convert to Spark DataFrame
                df = spark.createDataFrame(df_pandas)
                logger.info(f"✅ Successfully loaded CSV data from S3 - Shape: ({df.count()}, {len(df.columns)})")
            else:
                logger.warning(f"⚠️ Raw data not found in S3: {s3_key}, using local file")
                logger.info(f"💡 Run 'make s3-upload-data' to upload raw data to S3")
                ingestor = DataIngestorCSV(spark)
                df = ingestor.ingest(data_path)
        else:
            # Use local file
            logger.info(f"📁 Loading from local file: {data_path}")
            ingestor = DataIngestorCSV(spark)
            df = ingestor.ingest(data_path)
        logger.info(f"✓ Raw data loaded: {get_dataframe_info(df)}")
        
        # Log raw data metrics
        log_stage_metrics(df, 'raw', spark=spark)
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Handle missing values
        logger.info(f"\n{'='*80}")
        logger.info(f"HANDLING MISSING VALUES STEP")
        logger.info(f"{'='*80}")
        initial_count = df.count()
        
        # Drop critical missing values
        drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'], spark=spark)
        df = drop_handler.handle(df)
        
        # Fill Age column
        age_handler = FillMissingValuesStrategy(method='mean', relevant_column='Age', spark=spark)
        df = age_handler.handle(df)
        
        # Fill Gender column (skip API-based imputation for now, use simple fill)
        df = df.fillna({'Gender': 'Unknown'})
        
        rows_removed = initial_count - df.count()
        log_stage_metrics(df, 'missing_handled', {'rows_removed': rows_removed}, spark)
        logger.info(f"✓ Missing values handled: {initial_count} → {df.count()}")
        
        # Outlier detection
        logger.info(f"\n{'='*80}")
        logger.info(f"OUTLIER DETECTION STEP")
        logger.info(f"{'='*80}")
        initial_count = df.count()
        outlier_detector = OutlierDetector(strategy=IQROutlierDetection(spark=spark))
        df = outlier_detector.handle_outliers(df, columns['outlier_columns'], method='remove')
        
        outliers_removed = initial_count - df.count()
        log_stage_metrics(df, 'outliers_removed', {'outliers_removed': outliers_removed}, spark)
        logger.info(f"✓ Outliers removed: {initial_count} → {df.count()}")
        
        # Feature binning
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE BINNING STEP")
        logger.info(f"{'='*80}")
        binning = CustomBinningStrategy(binning_config['credit_score_bins'], spark=spark)
        df = binning.bin_feature(df, 'CreditScore')
        
        # Log binning distribution
        if 'CreditScoreBins' in df.columns:
            bin_dist = df.groupBy('CreditScoreBins').count().collect()
            bin_metrics = {f'credit_score_bin_{row["CreditScoreBins"]}': row['count'] for row in bin_dist}
            mlflow.log_metrics(bin_metrics)
        
        logger.info("✓ Feature binning completed")
        
        # Feature encoding
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE ENCODING STEP")
        logger.info(f"{'='*80}")
        nominal_strategy = NominalEncodingStrategy(encoding_config['nominal_columns'], spark=spark)
        ordinal_strategy = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'], spark=spark)
        
        df = nominal_strategy.encode(df)
        df = ordinal_strategy.encode(df)
        
        log_stage_metrics(df, 'encoded', spark=spark)
        logger.info("✓ Feature encoding completed")
        
        # Feature scaling
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE SCALING STEP")
        logger.info(f"{'='*80}")
        minmax_strategy = MinMaxScalingStrategy(spark=spark)
        df = minmax_strategy.scale(df, scaling_config['columns_to_scale'])
        logger.info("✓ Feature scaling completed")
        
        # Post-processing - drop unnecessary columns
        drop_columns = ['RowNumber', 'CustomerId', 'Firstname', 'Lastname']
        existing_drop_columns = [col for col in drop_columns if col in df.columns]
        if existing_drop_columns:
            df = df.drop(*existing_drop_columns)
            logger.info(f"✓ Dropped columns: {existing_drop_columns}")
        
        # Data splitting
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA SPLITTING STEP")
        logger.info(f"{'='*80}")
        splitting_strategy = SimpleTrainTestSplitStrategy(test_size=splitting_config['test_size'], spark=spark)
        X_train, X_test, Y_train, Y_test = splitting_strategy.split_data(df, target_column)
        
        # Save processed data
        output_paths = save_processed_data(X_train, X_test, Y_train, Y_test, pipeline_timestamp, output_format)
        
        logger.info("✓ Data splitting completed")
        logger.info(f"\nDataset shapes after splitting:")
        logger.info(f"  • X_train: {X_train.count()} rows, {len(X_train.columns)} columns")
        logger.info(f"  • X_test:  {X_test.count()} rows, {len(X_test.columns)} columns")
        logger.info(f"  • Y_train: {Y_train.count()} rows, 1 column")
        logger.info(f"  • Y_test:  {Y_test.count()} rows, 1 column")
        logger.info(f"  • Feature columns: {X_train.columns}")
        
        # Save preprocessing pipeline metadata to S3
        if hasattr(minmax_strategy, 'scaler_models'):
            # Save metadata about the preprocessing to S3
            preprocessing_metadata = {
                'scaling_columns': scaling_config['columns_to_scale'],
                'encoding_columns': encoding_config['nominal_columns'],
                'ordinal_mappings': encoding_config['ordinal_mappings'],
                'binning_config': binning_config,
                'spark_version': spark.version,
                'timestamp': pipeline_timestamp
            }
            
            # Try to save to S3, fallback to local if needed
            try:
                metadata_s3_key = f"artifacts/data_artifacts/{pipeline_timestamp}/preprocessing_metadata.json"
                from s3_io import put_bytes
                metadata_json = json.dumps(preprocessing_metadata, indent=2).encode('utf-8')
                put_bytes(metadata_json, key=metadata_s3_key, content_type='application/json')
                
                logger.info(f"✓ Saved preprocessing metadata to s3://{get_s3_bucket()}/{metadata_s3_key}")
                
            except Exception as s3_error:
                # Fallback to local save
                logger.warning(f"S3 metadata save failed ({s3_error}), saving locally")
                local_metadata_path = f'artifacts/encode/preprocessing_metadata_{pipeline_timestamp}.json'
                with open(local_metadata_path, 'w') as f:
                    json.dump(preprocessing_metadata, f, indent=2)
                logger.info(f"✓ Saved preprocessing metadata locally: {local_metadata_path}")
        
        # Final metrics and visualizations
        log_stage_metrics(X_train, 'final_train', spark=spark)
        log_stage_metrics(X_test, 'final_test', spark=spark)
        
        # Log comprehensive pipeline metrics
        comprehensive_metrics = {
            'total_samples': X_train.count() + X_test.count(),
            'train_samples': X_train.count(),
            'test_samples': X_test.count(),
            'final_features': len(X_train.columns),
            'processing_engine': 'pyspark',
            'output_format': output_format
        }
        
        # Get class distribution
        train_dist = Y_train.groupBy(target_column).count().collect()
        test_dist = Y_test.groupBy(target_column).count().collect()
        
        for row in train_dist:
            comprehensive_metrics[f'train_class_{row[target_column]}'] = row['count']
        for row in test_dist:
            comprehensive_metrics[f'test_class_{row[target_column]}'] = row['count']
        
        mlflow_tracker.log_data_pipeline_metrics(comprehensive_metrics)
        
        # Log parameters
        mlflow.log_params({
            'final_feature_names': X_train.columns,
            'preprocessing_steps': ['missing_values', 'outlier_detection', 'feature_binning', 
                                  'feature_encoding', 'feature_scaling'],
            'data_pipeline_version': '3.0_pyspark'
        })
        
        # Log artifacts
        for path_key, path_value in output_paths.items():
            if os.path.exists(path_value):
                mlflow.log_artifact(path_value, "processed_datasets")
        
        mlflow_tracker.end_run()
        
        # Convert to numpy arrays for return
        X_train_np = spark_to_pandas(X_train).values
        X_test_np = spark_to_pandas(X_test).values
        Y_train_np = spark_to_pandas(Y_train).values.ravel()
        Y_test_np = spark_to_pandas(Y_test).values.ravel()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL DATASET SHAPES")
        logger.info(f"{'='*80}")
        logger.info(f"✓ Final dataset shapes:")
        logger.info(f"  • X_train shape: {X_train_np.shape} (rows: {X_train_np.shape[0]}, features: {X_train_np.shape[1]})")
        logger.info(f"  • X_test shape:  {X_test_np.shape} (rows: {X_test_np.shape[0]}, features: {X_test_np.shape[1]})")
        logger.info(f"  • Y_train shape: {Y_train_np.shape} (rows: {Y_train_np.shape[0]})")
        logger.info(f"  • Y_test shape:  {Y_test_np.shape} (rows: {Y_test_np.shape[0]})")
        logger.info(f"  • Total samples: {X_train_np.shape[0] + X_test_np.shape[0]}")
        logger.info(f"  • Train/Test ratio: {X_train_np.shape[0]/(X_train_np.shape[0] + X_test_np.shape[0]):.1%} / {X_test_np.shape[0]/(X_train_np.shape[0] + X_test_np.shape[0]):.1%}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info("✓ PySpark data pipeline completed successfully!")
        
        return {
            'X_train': X_train_np,
            'X_test': X_test_np,
            'Y_train': Y_train_np,
            'Y_test': Y_test_np
        }
        
    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {str(e)}")
        if 'mlflow_tracker' in locals():
            mlflow_tracker.end_run()
        raise
    finally:
        # Stop Spark session
        stop_spark_session(spark)


if __name__ == "__main__":
    # Run the pipeline
    processed_data = data_pipeline(output_format="csv")
    logger.info(f"Pipeline completed. Train samples: {processed_data['X_train'].shape[0]}")