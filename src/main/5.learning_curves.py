# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules
from models.train_model import *
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
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import lightgbm as lgb

# Extract
seed(40)
train = read_csv("../../data/interim/train.csv")
train_y = train[["y"]].values
dev = read_csv("../../data/interim/dev.csv")
dev_y = dev[["y"]].values

# Data parameters
features_pipeline = data_preparation()

# Model parameters
full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", LogisticRegression(random_state=1)),
    # ("clf", lgb.LGBMClassifier(class_weight='balanced')),
])

# Learning curve
train_sizes = (np.linspace(0.1, 1.0, 10)*train.shape[0]).astype(int)
training_scores, dev_scores = [], []
for train_size in train_sizes:
    temp_train = train.iloc[0:train_size]
    temp_train_y = temp_train[["y"]].values
    full_pipeline.fit(temp_train, temp_train_y)
    precision_threshold = 0.20
    train_prob_y = full_pipeline.predict_proba(train)[:, 1]
    dev_prob_y = full_pipeline.predict_proba(dev)[:, 1]
    train_precision, train_recall, _ = precision_recall_curve(train_y, train_prob_y, pos_label=1)
    dev_precision, dev_recall, _ = precision_recall_curve(dev_y, dev_prob_y, pos_label=1)
    training_scores.append(max([y for (x,y) in zip(train_precision, train_recall) if x >= precision_threshold]))
    dev_scores.append(max([y for (x,y) in zip(dev_precision, dev_recall) if x >= precision_threshold]))


plt.plot(train_sizes, training_scores,
         color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, dev_scores,
         color='green', marker='o', markersize=5, label='dev accuracy')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.ylim([0.0, 1.0])
plt.savefig("../../reports/figures/lr_learning_curve.png")
plt.show()
