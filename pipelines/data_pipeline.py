import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStratergy
from feature_encoding import OrdinalEncodingStratergy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStratergy
from data_spiltter import SimpleTrainTestSplitStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
import mlflow


def create_data_visualizations(df: pd.DataFrame, stage: str, artifacts_dir: str):
    """Create essential data visualizations for MLflow artifacts."""
    try:
        stage_dir = os.path.join(artifacts_dir, f"visualizations_{stage}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # 1. Data distribution for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:4]):  # Top 4 numeric columns
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(numeric_cols), 4):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Data Distributions - {stage.title()}')
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f'distributions_{stage}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation heatmap for numeric features
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title(f'Feature Correlation - {stage.title()}')
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f'correlation_{stage}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Log visualizations to MLflow
        for viz_file in os.listdir(stage_dir):
            if viz_file.endswith('.png'):
                mlflow.log_artifact(os.path.join(stage_dir, viz_file), f"visualizations/{stage}")
        
        logger.info(f"Visualizations created for {stage}")
        
    except Exception as e:
        logger.error(f"Failed to create visualizations for {stage}: {str(e)}")


def log_stage_metrics(df: pd.DataFrame, stage: str, additional_metrics: Dict = None):
    """Log key metrics for each processing stage."""
    try:
        metrics = {
            f'{stage}_rows': df.shape[0],
            f'{stage}_columns': df.shape[1],
            f'{stage}_missing_values': df.isnull().sum().sum(),
            f'{stage}_memory_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        if additional_metrics:
            metrics.update({f'{stage}_{k}': v for k, v in additional_metrics.items()})
        
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics logged for {stage}: {df.shape}")
        
    except Exception as e:
        logger.error(f"Failed to log metrics for {stage}: {str(e)}")


def log_csv_artifacts(csv_files: Dict[str, str], artifacts_dir: str):
    """Log final CSV files as MLflow artifacts with metadata."""
    try:
        csv_metadata = {
            'csv_files': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create CSV artifacts directory
        csv_artifacts_dir = os.path.join(artifacts_dir, 'final_csv_files')
        os.makedirs(csv_artifacts_dir, exist_ok=True)
        
        total_files_logged = 0
        
        for file_name, file_path in csv_files.items():
            if os.path.exists(file_path):
                try:
                    # Get file metadata
                    file_size = os.path.getsize(file_path) / (1024**2)  # MB
                    df = pd.read_csv(file_path)
                    
                    csv_metadata['csv_files'][file_name] = {
                        'file_path': file_path,
                        'file_size_mb': round(file_size, 2),
                        'shape': df.shape,
                        'columns': list(df.columns) if len(df.columns) <= 20 else f"{len(df.columns)} columns",
                        'sample_values': df.head(2).to_dict() if df.shape[0] > 0 else "No data"
                    }
                    
                    # Log the CSV file as artifact
                    mlflow.log_artifact(file_path, "final_datasets")
                    
                    # Log key metrics
                    mlflow.log_metrics({
                        f'final_{file_name}_rows': df.shape[0],
                        f'final_{file_name}_columns': df.shape[1],
                        f'final_{file_name}_size_mb': file_size
                    })
                    
                    total_files_logged += 1
                    logger.info(f"Logged {file_name}: {df.shape} ({file_size:.2f}MB)")
                    
                except Exception as e:
                    logger.warning(f"Could not process {file_name}: {str(e)}")
                    csv_metadata['csv_files'][file_name] = {
                        'file_path': file_path,
                        'error': str(e)
                    }
            else:
                logger.warning(f"File not found: {file_path}")
                csv_metadata['csv_files'][file_name] = {
                    'file_path': file_path,
                    'status': 'not_found'
                }
        
        # Save CSV metadata
        metadata_path = os.path.join(csv_artifacts_dir, 'final_csv_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(csv_metadata, f, indent=2, default=str)
        
        # Log metadata as artifact
        mlflow.log_artifact(metadata_path, "final_datasets")
        
        # Log summary metrics
        mlflow.log_metrics({
            'total_csv_files_logged': total_files_logged,
            'csv_artifacts_created': len(csv_files)
        })
        
        logger.info(f"CSV artifacts logged: {total_files_logged}/{len(csv_files)} files")
        
    except Exception as e:
        logger.error(f"Failed to log CSV artifacts: {str(e)}")


def data_pipeline(
                data_path: str = 'data/raw/ChurnModelling.csv',
                target_column: str = 'Exited',
                test_size: float = 0.2,
                force_rebuild: bool = False
                ) -> Dict[str, np.ndarray]:

    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()

    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()
    run_tags = create_mlflow_run_tags(
                                    'data_pipeline', {
                                        'data_source' : data_path
                                    }
                                    )
    run = mlflow_tracker.start_run(run_name='data_pipeliine', tags=run_tags)


    print('Step 01 : Data Ingestion')
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
    x_train_path = os.path.join('artifacts', 'data', 'X_train.csv')
    x_test_path = os.path.join('artifacts', 'data', 'X_test.csv')
    y_train_path = os.path.join('artifacts', 'data', 'Y_train.csv')
    y_test_path = os.path.join('artifacts', 'data', 'Y_test.csv')
    
    if os.path.exists(x_train_path) and \
        os.path.exists(x_test_path) and \
        os.path.exists(y_train_path) and \
        os.path.exists(y_test_path):
        
        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        Y_train = pd.read_csv(y_train_path)
        Y_test = pd.read_csv(y_test_path) 


    os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)
    if not os.path.exists('temp_imputed.csv'):
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)
        print(f"loaded data shape : {df.shape}")

        print('\nStep 02 : Handle Missing Values')
        drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'])

        age_handler = FillMissingValuesStrategy(
                                                method='mean',
                                                relevent_column='Age'
                                                )
        
        gender_handler = FillMissingValuesStrategy(
                                                    relevent_column=None,
                                                    is_custom_imputer=True,
                                                    custom_imputer=GenderImputer()                                                                                      
                                                    )
        df = drop_handler.handle(df)
        df = age_handler.handle(df)
        df = gender_handler.handle(df)
        df.to_csv('temp_imputed.csv', index=False)

    df = pd.read_csv('temp_imputed.csv')

    print(f"data shape after inputation : {df.shape}")

    print('\nStep 03: Handle Outliers')

    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
    df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
    print(f'data shape after outlier removal : {df.shape}')


    print('\nStep 04 : Feature Bining')

    binning = CustomBinningStratergy(binning_config['credit_score_bins'])
    df = binning.bin_feature(df, 'CreditScore')
    print(f"data after feature binning : \n{df.head()}")

    print('\nStep 05 : Feature Encoding')

    nominal_strategy = NominalEncodingStrategy(encoding_config['nominal_columns'])
    ordinal_strategy = OrdinalEncodingStratergy(encoding_config['ordinal_mappings'])

    df = nominal_strategy.encode(df)
    df = ordinal_strategy.encode(df)
    print(f"data after feature encoding : \n{df.head()}")
    
    print('\nStep 06 : Feature Scaling')
    minmax_strategy = MinMaxScalingStratergy()
    df = minmax_strategy.scale(df, scaling_config['columns_to_scale'])
    print(f"data after feature scaling : \n{df.head()}")

    print('\nStep 07 : Post Processing ')
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Firstname', 'Lastname'])
    print(f'data after post processing : \n {df.head()}')

    print('\nStep 08 : Data Splitting')
    splitting_stratergy = SimpleTrainTestSplitStratergy(test_size=splitting_config['test_size'])
    X_train, X_test, Y_train, Y_test = splitting_stratergy.split_data(df, 'Exited')

    # Create directories and save splits
    os.makedirs('artifacts/data', exist_ok=True)
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    Y_train.to_csv(y_train_path, index=False)
    Y_test.to_csv(y_test_path, index=False)

    print(f'X train size : {X_train.shape}')
    print(f'X test size : {X_test.shape}')
    print(f'Y train size : {Y_train.shape}')
    print(f'Y test size : {Y_test.shape}')

    mlflow_tracker.log_data_pipeline_metrics({
                                            'total_samples' : len(X_train) + len(X_test),
                                            'train_samples' : len(X_train),
                                            'test_samples' : len(X_test),
                                            'x_train_path' : x_train_path,
                                            'x_test_path' : x_test_path,
                                            'y_train_path' : y_train_path,
                                            'y_test_path' : y_test_path
                                            })
    mlflow_tracker.end_run()


# data_pipeline()           

        
            