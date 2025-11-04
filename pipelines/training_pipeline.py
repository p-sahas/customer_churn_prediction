import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_pipeline import data_pipeline
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_building import XGboostModelBuilder, RandomForestModelBuilder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
#from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
from config import get_model_config, get_data_paths
#import mlflow
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

