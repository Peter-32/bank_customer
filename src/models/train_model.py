import warnings
warnings.filterwarnings("ignore")

# Public modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def train_model(features_pipeline):
    train = read_csv("../../data/interim/train.csv")
    train_y = train[["y"]].values
    full_pipeline = Pipeline([
        ("features", features_pipeline),
        ("clf", LogisticRegression(random_state=1)),
    ])

    scores = cross_val_score(estimator=full_pipeline, X=train, y=train_y,
                             cv=10, n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
