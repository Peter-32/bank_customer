# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Modify train and test set
dataset_modifications_after_exploring_data()

# Peek at the data
train = read_csv("../../data/interim/train_v2.csv")
test = read_csv("../../data/interim/test_v2.csv")
#
#
#
# df = df.drop(["duration"], axis=1)
#
# # Find missing values
# any_missing = df.isna().any(); missing_attributes = list(any_missing[any_missing == True].index)
# df['y'] = df['y'].apply(lambda x: 1 if x == "yes" else 0)
# print("Missing values are in these attributes:", missing_attributes)
#
#
#
#
# # Split data
# train, sample_train, test = split_data(df=df)
