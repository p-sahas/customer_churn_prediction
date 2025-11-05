import os
import sys
import json
import pandas as pd
import logging
import numpy as np
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_inference_config


inference = ModelInference('artifacts/models/churn_analysis.joblib')


def streaming_inference(inference, data):
    inference.load_encoders('artifacts/encode')
    pred = inference.predict(data)
    return pred

if __name__ == '__main__':
    data = {
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
    pred = streaming_inference(inference, data)
    print(pred)