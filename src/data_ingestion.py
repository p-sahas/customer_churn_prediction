import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union
import pandas as pd  # Keep pandas import for educational purposes
from pyspark.sql import DataFrame, SparkSession
from utils.spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    """Abstract base class for data ingestion supporting both pandas and PySpark."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize DataIngestor with a SparkSession.
        
        Args:
            spark: Optional SparkSession. If not provided, will create/get one.
        """
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> DataFrame:
        """
        Ingest data from the specified path.
        
        Args:
            file_path_or_link: Path to the data file
            
        Returns:
            DataFrame (PySpark or pandas depending on implementation)
        """
        pass


class DataIngestorCSV(DataIngestor):
    """CSV data ingestion implementation."""
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        """
        Ingest CSV data using PySpark.
        
        Args:
            file_path_or_link: Path to the CSV file
            **options: Additional options for CSV reading
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - CSV (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting CSV data ingestion from: {file_path_or_link}")
        
        # Define S3 constants
        from config import get_s3_bucket
        S3_BUCKET = get_s3_bucket()
        S3_KEY = "data/raw/ChurnModelling.csv"
        S3A_URI = f"s3a://{S3_BUCKET}/{S3_KEY}"
        LOCAL_PATH = "data/raw/ChurnModelling.csv"
        
        try:
            # Check if we should load from S3A or local file
            if file_path_or_link.startswith('s3://'):
                # Convert s3:// to s3a:// for Spark compatibility
                data_path = file_path_or_link.replace('s3://', 's3a://')
                logger.info(f"📁 Loading from S3A (converted from s3://): {data_path}")
            elif file_path_or_link.startswith('s3a://'):
                # Already S3A format
                data_path = file_path_or_link
                logger.info(f"📁 Loading from S3A: {data_path}")
            else:
                # Handle local path or S3 URL from config
                import sys
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
                from config import get_s3_bucket, force_s3_io
                from s3_io import key_exists
                
                # Check if this is an S3 URL that needs conversion
                if 's3://' in file_path_or_link:
                    # Extract bucket and key from full S3 URL to avoid double prefixing
                    s3_url_parts = file_path_or_link.replace('s3://', '').split('/', 1)
                    if len(s3_url_parts) == 2:
                        bucket, key = s3_url_parts
                        use_path = f"s3a://{bucket}/{key}"
                        
                        # Check if S3 object exists (avoid double prefixing)
                        try:
                            if key_exists(key):
                                data_path = use_path
                                logger.info(f"📁 S3 object exists, using S3A: {data_path}")
                            else:
                                logger.warning(f"⚠️ S3 object not found: {key}, falling back to local")
                                data_path = LOCAL_PATH
                        except Exception as s3_check_error:
                            logger.warning(f"⚠️ S3 check failed: {s3_check_error}, using local")
                            data_path = LOCAL_PATH
                    else:
                        logger.warning("⚠️ Invalid S3 URL format, using local")
                        data_path = LOCAL_PATH
                else:
                    # Try S3A first if force_s3_io is enabled
                    if force_s3_io():
                        try:
                            if key_exists(S3_KEY):
                                data_path = S3A_URI
                                logger.info(f"📁 S3 object exists, using S3A: {data_path}")
                            else:
                                logger.warning(f"⚠️ S3 key not found: {S3_KEY}, falling back to local")
                                data_path = LOCAL_PATH
                        except Exception as s3_error:
                            logger.warning(f"⚠️ S3 access failed: {s3_error}, using local")
                            data_path = LOCAL_PATH
                    else:
                        data_path = file_path_or_link
                        logger.info(f"📁 Loading from local file: {data_path}")
            
            # Default CSV options
            csv_options = {
                "header": "true",
                "inferSchema": "true",
                "ignoreLeadingWhiteSpace": "true",
                "ignoreTrailingWhiteSpace": "true",
                "nullValue": "",
                "nanValue": "NaN",
                "escape": '"',
                "quote": '"'
            }
            csv_options.update(options)
            
            ############### PYSPARK CODES ###########################
            # Read CSV file (from S3 or local)
            df = self.spark.read.options(**csv_options).csv(data_path)
            
            # Get DataFrame info
            row_count = df.count()
            columns = df.columns
            
            # Calculate approximate memory usage
            # Note: This is an estimate as PySpark distributes data
            sample_size = min(1000, row_count)
            if row_count > 0:
                sample_df = df.limit(sample_size).toPandas()
                memory_per_row = sample_df.memory_usage(deep=True).sum() / sample_size
                estimated_memory = (memory_per_row * row_count) / 1024**2
            else:
                estimated_memory = 0
            
            ############### PANDAS CODES ###########################
            # logger.info(f"✓ Successfully loaded CSV data - Shape: {df.shape}, Columns: {list(df.columns)}")
            # logger.info(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            ############### PYSPARK CODES ###########################
            logger.info(f"✓ Successfully loaded CSV data - Shape: ({row_count}, {len(columns)})")
            logger.info(f"✓ Columns: {columns}")
            logger.info(f"✓ Estimated memory usage: {estimated_memory:.2f} MB")
            logger.info(f"✓ Partitions: {df.rdd.getNumPartitions()}")
            
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load CSV data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorExcel(DataIngestor):
    """Excel data ingestion implementation."""
    
    def ingest(self, file_path_or_link: str, sheet_name: Optional[str] = None, **options) -> DataFrame:
        """
        Ingest Excel data using PySpark.
        Note: This implementation converts Excel to CSV format internally as PySpark
        doesn't have native Excel support. For production use, consider using
        spark-excel library.
        
        Args:
            file_path_or_link: Path to the Excel file
            sheet_name: Name of the sheet to read (optional)
            **options: Additional options
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - EXCEL (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Excel data ingestion from: {file_path_or_link}")
        
        try:
            # For Excel files, we need to use pandas as an intermediary
            # In production, consider using spark-excel library
            logger.info("⚠ Note: Using pandas for Excel reading, then converting to PySpark")
            
            ############### PANDAS CODES ###########################
            # df = pd.read_excel(file_path_or_link)
            
            ############### PYSPARK CODES ###########################
            # Read Excel with pandas first
            pandas_df = pd.read_excel(file_path_or_link, sheet_name=sheet_name)
            
            # Convert to PySpark DataFrame
            df = self.spark.createDataFrame(pandas_df)
            
            # Get DataFrame info
            row_count = df.count()
            columns = df.columns
            
            ############### PANDAS CODES ###########################
            # logger.info(f"✓ Successfully loaded Excel data - Shape: {df.shape}, Columns: {list(df.columns)}")
            # logger.info(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            ############### PYSPARK CODES ###########################
            logger.info(f"✓ Successfully loaded Excel data - Shape: ({row_count}, {len(columns)})")
            logger.info(f"✓ Columns: {columns}")
            logger.info(f"✓ Partitions: {df.rdd.getNumPartitions()}")
            
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load Excel data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorParquet(DataIngestor):
    """PySpark Parquet data ingestion implementation (new for PySpark)."""
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        """
        Ingest Parquet data using PySpark.
        Note: Parquet is a columnar format optimized for big data processing.
        
        Args:
            file_path_or_link: Path to the Parquet file or directory
            **options: Additional options for Parquet reading
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - PARQUET (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Parquet data ingestion from: {file_path_or_link}")
        
        try:
            # Read Parquet file(s)
            df = self.spark.read.options(**options).parquet(file_path_or_link)
            
            # Get DataFrame info
            row_count = df.count()
            columns = df.columns
            
            # Parquet files are already compressed and optimized
            logger.info(f"✓ Successfully loaded Parquet data - Shape: ({row_count}, {len(columns)})")
            logger.info(f"✓ Columns: {columns}")
            logger.info(f"✓ Partitions: {df.rdd.getNumPartitions()}")
            logger.info(f"✓ Schema: {df.schema.simpleString()}")
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load Parquet data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorFactory:
    """Factory class to create appropriate data ingestor based on file type."""
    
    @staticmethod
    def get_ingestor(file_path: str, spark: Optional[SparkSession] = None) -> DataIngestor:
        """
        Get appropriate data ingestor based on file extension.
        
        Args:
            file_path: Path to the data file
            spark: Optional SparkSession
            
        Returns:
            DataIngestor: Appropriate ingestor instance
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return DataIngestorCSV(spark)
        elif file_extension in ['.xlsx', '.xls']:
            return DataIngestorExcel(spark)
        elif file_extension == '.parquet':
            return DataIngestorParquet(spark)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


def smoke_test_s3a_read():
    """Smoke test for S3A CSV reading"""
    # Define S3 constants
    from utils.config import get_s3_bucket
    S3_BUCKET = get_s3_bucket()
    S3_KEY = "data/raw/ChurnModelling.csv"
    S3A_URI = f"s3a://{S3_BUCKET}/{S3_KEY}"
    LOCAL_PATH = "data/raw/ChurnModelling.csv"
    
    print("🧪 Testing S3A CSV reading...")
    
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
        from spark_session import create_spark_session
        from s3_io import key_exists
        
        # Create Spark session with S3A support
        spark = create_spark_session("S3A_SmokeTest")
        
        # Check which path to test
        try:
            if key_exists(S3_KEY):
                test_path = S3A_URI
                print(f"✅ S3 object exists, testing S3A: {test_path}")
            else:
                test_path = LOCAL_PATH
                print(f"⚠️ S3 object not found, testing local: {test_path}")
        except Exception:
            test_path = LOCAL_PATH
            print(f"⚠️ S3 check failed, testing local: {test_path}")
        
        # Test read
        df = spark.read.option("header", "true").csv(test_path).limit(1)
        result = df.collect()
        
        print(f"✅ S3A smoke test passed! Read {len(result)} row(s)")
        print(f"📊 Columns: {df.columns}")
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"❌ S3A smoke test failed: {e}")
        try:
            spark.stop()
        except:
            pass
        return False


if __name__ == "__main__":
    smoke_test_s3a_read()