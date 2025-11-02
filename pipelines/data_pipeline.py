import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
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
        df.to_csv('temp_imputed.csv')

    df = pd.read_csv('temp_imputed.csv')

    print(f"data shape after inputation : {df.shape}")

    print('\nStep 03: Handle Outliers')

    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
    df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
    print(f'data shape after outlier removal : {df.shape}')
data_pipeline()            

        
            