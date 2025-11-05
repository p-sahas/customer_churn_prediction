import json
import logging
import os
import joblib, sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from feature_binning import CustomBinningStratergy
from feature_encoding import OrdinalEncodingStratergy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config
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
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.encoders = {}

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise ValueError("Can't load. Model not found.")
        
        self.model = joblib.load(self.model_path)

    def load_encoders(self, encoders_dir):
        for file in os.listdir(encoders_dir):
            feature_name = file.split('_encoder.json')[0]
            with open(os.path.join(encoders_dir, file), 'r') as f:
                self.encoders[feature_name] = json.load(f)

    def preprocess_input(self, data):
        data = pd.DataFrame([data])


        for col, encoder in self.encoders.items():
            data[col] = data[col].map(encoder)

            return data
        
data = {
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
  "Exited": 1
}

inference = ModelInference('artifacts/models/churn_analysis.joblib')
inference.load_encoders('artifacts/encode')

data = inference.preprocess_input(data)
print(data)