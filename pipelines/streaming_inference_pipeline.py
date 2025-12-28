#!/usr/bin/env python3
"""
Batch Inference Pipeline

This pipeline performs batch inference on processed data using the trained model.
Replaces the streaming inference pipeline with a simpler batch approach.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import project modules
from src.model_inference import ModelInference
from utils.spark_session import create_spark_session
from utils.config import load_config, get_s3_bucket, force_s3_io
from utils.mlflow_utils import MLflowTracker
from utils.s3_artifact_manager import S3ArtifactManager, get_latest_s3_artifacts
from utils.s3_io import read_df_csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchInferencePipeline:
    """Batch inference pipeline for ML predictions"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the batch inference pipeline"""
        self.config = load_config()
        self.model_inference = None
        self.spark = None
        self.mlflow_tracker = MLflowTracker()
        
        # Generate single timestamp for entire inference pipeline run
        from datetime import datetime
        self.pipeline_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        logger.info(f" Inference pipeline timestamp: {self.pipeline_timestamp}")
        
    def initialize(self):
        """Initialize components"""
        try:
            logger.info(" Initializing Batch Inference Pipeline")
            logger.info("=" * 60)
            
            # Initialize Spark session
            self.spark = create_spark_session("BatchInferencePipeline")
            logger.info("✅ Spark session initialized")
            
            # Initialize model inference
            # Use MLflow model registry instead of S3 path
            model_path = "spark_random_forest_model"  # This will trigger latest model search
            self.model_inference = ModelInference(
                model_path=model_path, 
                use_spark=True, 
                spark=self.spark
            )
            logger.info("✅ Model inference initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    def load_test_data(self, sample_size: int = 1000) -> pd.DataFrame:
        """Load test data for batch inference with random sampling"""
        try:
            logger.info(f" Loading test data for inference (sampling {sample_size} records)...")
            
            # Try to load latest test data from S3
            s3_manager = S3ArtifactManager()
            try:
                latest_paths = s3_manager.get_latest_artifacts(['X_test'], artifact_type='data_artifacts', format_ext='csv')
                if 'X_test' in latest_paths:
                    test_s3_key = latest_paths['X_test']
                    logger.info(f" Using latest S3 artifact: s3://{get_s3_bucket()}/{test_s3_key}")
                    df = read_df_csv(key=test_s3_key)
                    
                    # Randomly sample the specified number of records
                    original_size = len(df)
                    if len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42)
                        logger.info(f" Randomly sampled {sample_size} records from {original_size} total records")
                    else:
                        logger.info(f" Using all {len(df)} available records (less than requested {sample_size})")
                    
                    logger.info(f"✅ Loaded test data: {df.shape}")
                    return df
            except Exception as e:
                logger.warning(f"⚠️ Could not load latest S3 artifacts: {e}")
            
            # Fallback to legacy local path (if S3 is not enforced)
            if not force_s3_io():
                test_data_path = Path("data/artifacts/csv/latest/X_test.csv")
                if test_data_path.exists():
                    logger.info(" Using legacy local path (S3 not enforced)")
                    df = pd.read_csv(test_data_path)
                    
                    # Randomly sample the specified number of records
                    original_size = len(df)
                    if len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42)
                        logger.info(f" Randomly sampled {sample_size} records from {original_size} total records")
                    else:
                        logger.info(f" Using all {len(df)} available records (less than requested {sample_size})")
                    
                    logger.info(f"✅ Loaded test data: {df.shape}")
                    return df
            else:
                # Load raw data and take a sample for inference
                logger.info(" No processed test data found, using raw data")
                raw_data_path = Path("data/raw/ChurnModelling.csv")
                if raw_data_path.exists():
                    df = pd.read_csv(raw_data_path)
                    # Remove target column if present
                    if 'Exited' in df.columns:
                        df = df.drop('Exited', axis=1)
                    
                    # Take a random sample for inference
                    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
                    logger.info(f" Randomly sampled {len(sample_df)} records from raw data ({len(df)} total)")
                    logger.info(f"✅ Loaded sample data for inference: {sample_df.shape}")
                    return sample_df
                else:
                    raise FileNotFoundError("No data available for inference")
                    
        except Exception as e:
            logger.error(f"❌ Error loading test data: {str(e)}")
            raise
    
    def run_batch_inference(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run batch inference on the data"""
        try:
            logger.info(" Running batch inference...")
            logger.info(f" Processing {len(data)} records")
            
            predictions = []
            successful_predictions = 0
            
            # Process each record
            for idx, row in data.iterrows():
                try:
                    # Convert row to dictionary
                    record_dict = row.to_dict()
                    
                    # Make prediction
                    prediction = self.model_inference.predict(record_dict)
                    
                    # Combine original data with prediction
                    result = {
                        **record_dict,
                        **prediction,
                        'processed_at': datetime.now().isoformat(),
                        'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'record_index': idx
                    }
                    
                    predictions.append(result)
                    successful_predictions += 1
                    
                    if successful_predictions % 25 == 0:
                        logger.info(f" Processed {successful_predictions}/{len(data)} records")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process record {idx}: {str(e)}")
                    continue
            
            # Convert to DataFrame
            results_df = pd.DataFrame(predictions)
            
            logger.info(f"✅ Batch inference completed")
            logger.info(f" Successfully processed: {successful_predictions}/{len(data)} records")
            
            return results_df
            
        except Exception as e:
            logger.error(f"❌ Error in batch inference: {str(e)}")
            raise
    
    def save_results(self, results_df: pd.DataFrame):
        """Save inference results to S3"""
        try:
            logger.info(" Saving inference results to S3...")
            
            # Save results to S3
            bucket = get_s3_bucket()
            
            # Save detailed results using pipeline timestamp
            results_key = f"artifacts/inference_artifacts/{self.pipeline_timestamp}/inference_results.json"
            from utils.s3_io import write_df_json
            write_df_json(results_df, key=results_key)
            logger.info(f"✅ Results saved to: s3://{bucket}/{results_key}")
            
            # Save summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_records': len(results_df),
                's3_key': results_key,
                'sample_predictions': results_df.head(3).to_dict('records') if len(results_df) > 0 else []
            }
            
            # Save summary to S3
            summary_key = f"artifacts/inference_artifacts/{self.pipeline_timestamp}/inference_summary.json"
            from s3_io import put_bytes
            import json
            summary_json = json.dumps(summary, indent=2).encode('utf-8')
            put_bytes(summary_json, key=summary_key, content_type='application/json')
            
            logger.info(f"✅ Summary saved to: s3://{bucket}/{summary_key}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete batch inference pipeline"""
        try:
            logger.info(" STARTING BATCH INFERENCE PIPELINE")
            logger.info("=" * 60)
            
            # Start MLflow run
            run = self.mlflow_tracker.start_run("batch_inference_pipeline")
            
            # Initialize components
            if not self.initialize():
                raise Exception("Failed to initialize pipeline")
            
            # Load test data (sample 1000 records)
            test_data = self.load_test_data(sample_size=1000)
            
            # Run inference
            results_df = self.run_batch_inference(test_data)
            
            # Save results
            self.save_results(results_df)
            
            # Log metrics to MLflow
            metrics = {
                'total_records': len(test_data),
                'successful_predictions': len(results_df),
                'success_rate': len(results_df) / len(test_data) if len(test_data) > 0 else 0
            }
            
            self.mlflow_tracker.log_inference_metrics(
                predictions=results_df['prediction'].values if 'prediction' in results_df.columns else None,
                input_data_info=metrics
            )
            
            logger.info(" Batch inference pipeline completed successfully!")
            logger.info(f" Success rate: {metrics['success_rate']:.2%}")
            
            # End MLflow run
            self.mlflow_tracker.end_run()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            if 'run' in locals():
                self.mlflow_tracker.end_run()
            return False
        finally:
            if self.spark:
                self.spark.stop()
                logger.info("🔚 Spark session stopped")


def main():
    """Main function"""
    try:
        pipeline = BatchInferencePipeline()
        success = pipeline.run_pipeline()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    import json
    exit(main())
