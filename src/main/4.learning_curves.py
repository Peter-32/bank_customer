# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

import warnings
warnings.filterwarnings("ignore")

# My modules
from models.train_model import *
from features.build_features import *

features_pipeline = data_preparation()
learning_curves(features_pipeline)
