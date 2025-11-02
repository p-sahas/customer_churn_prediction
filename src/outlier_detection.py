import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df, columns):
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            logging.info('Outliers detected using IQR Method.')
            return outliers
        
class OutlierDetector:
    def __init__(self, stratergy):
        self.strategy = stratergy

    def detect_outliers(self, df, selected_columns):
        return self._stratergy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df, selected_columns, method='remove'):
        outliers = self.detect_outliers(self, df, selected_columns)
        outlier_count = outliers.sum(axis=1) # Getting for each row
        rows_to_remove = outlier_count >= 2
        return df[~rows_to_remove]
    


