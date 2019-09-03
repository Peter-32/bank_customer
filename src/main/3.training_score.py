# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules
from features.build_features import *

# Public modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, \
                            precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from numpy.random import seed

# Inputs
SHOW_ERROR_ANALYSIS = False

# Extract
seed(40)
train = read_csv("../../data/interim/train.csv")
train_y = train[["y"]].values

# Data parameters
features_pipeline = data_preparation()

# Model parameters
full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", LogisticRegression(random_state=1)),
])

# Fit
full_pipeline.fit(train, train_y)

# Predict
precision_threshold = 0.20
prob_y = full_pipeline.predict_proba(train)[:, 1]
precision, recall, _ = precision_recall_curve(train_y, prob_y, pos_label=1)
score = max([y for (x,y) in zip(precision, recall) if x >= precision_threshold])
print('Recall score: %.3f' % score)


# Error analysis
if SHOW_ERROR_ANALYSIS:
    precision_threshold_index = min([i for (x,i) in zip(precision, range(len(precision))) if x >= precision_threshold])
    train["prob_y"] = prob_y
    prob_y_threshold = train.sort_values("prob_y", ascending=False).iloc[len(precision) - precision_threshold_index].prob_y
    pred_y = (prob_y >= prob_y_threshold).astype(bool)
    print("Prob y Threshold: %.1f" % (prob_y_threshold*100))
    print(confusion_matrix(train_y, pred_y))
    print("Recall: %.1f" % (recall_score(train_y, pred_y)*100))
    print("Precision: %.1f" % (precision_score(train_y, pred_y)*100))
