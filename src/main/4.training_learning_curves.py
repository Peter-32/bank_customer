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

# Data parameters
features_pipeline = data_preparation()

# Model parameters
full_pipeline = Pipeline([
    ("features", features_pipeline),
    # ("clf", LogisticRegression(random_state=1)),
    ("clf", lgb.LGBMClassifier(class_weight='balanced')),
])

# TESTING THIS NEW CODE:
# TESTING THIS NEW CODE:
# TESTING THIS NEW CODE:
# TESTING THIS NEW CODE:
def custom_cv_2folds(X):
    n = X.shape[0]
    i = 1
    while i <= 2:
        if i == 1:
            idx = np.arange(n * (i - 1) / 2, n-15, dtype=int)
            yield idx, idx
        if i == 2:
            idx = np.arange(n - 15, n, dtype=int)
            yield idx, idx
        i += 1
custom_cv = custom_cv_2folds(train)

# Learning curve
train_sizes = (np.linspace(0.1, 1.0, 10)*train.shape[0]).astype(int)
training_scores = []
for train_size in train_sizes:
    temp_train = train.iloc[0:train_size]
    temp_train_y = temp_train[["y"]].values
    full_pipeline.fit(temp_train, temp_train_y)
    precision_threshold = 0.20
    prob_y = full_pipeline.predict_proba(temp_train)[:, 1]
    precision, recall, _ = precision_recall_curve(temp_train_y, prob_y, pos_label=1)
    score = max([y for (x,y) in zip(precision, recall) if x >= precision_threshold])
    training_scores.append(score)

plt.plot(train_sizes, training_scores,
         color='blue', marker='o', markersize=5, label='training accuracy')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.ylim([0.0, 1.0])
plt.savefig("../../reports/figures/training_learning_curve.png")
plt.show()
