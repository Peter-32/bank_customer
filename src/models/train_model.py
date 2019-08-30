# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

import warnings
warnings.filterwarnings("ignore")

# My modules
from models.predict_model import *

# Public modules
import numpy as np
from pandas import read_csv
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve

def _common_code():
    seed(40)
    train = read_csv("../../data/interim/train.csv")
    train_y = train[["y"]].values
    full_pipeline = Pipeline([
        ("features", features_pipeline),
        ("clf", LogisticRegression(random_state=1)),
    ])
    return full_pipeline, train, train_y


def train_model(features_pipeline):
    full_pipeline, train, train_y = _common_code()
    get_score(full_pipeline, train, train_y)


def learning_curves(features_pipeline):
    full_pipeline, train, train_y = _common_code()
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator=full_pipeline,
                       X=train,
                       y=train_y,
                       train_sizes=np.linspace(0.1, 1.0, 10),
                       cv=10,
                       n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.875, 0.925])
    plt.show()
