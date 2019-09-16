# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules
from features.build_features import *

# Public modules
from sklearn.model_selection import GridSearchCV
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
import lightgbm as lgb

# Inputs
SHOW_ERROR_ANALYSIS = True

# Extract
seed(40)
train = read_csv("../../data/interim/train.csv")
train_y = train[["y"]].values
dev = read_csv("../../data/interim/dev.csv")
dev_y = dev[["y"]].values
test = read_csv("../../data/interim/test.csv")
test_y = test[["y"]].values

# Data parameters
features_pipeline = data_preparation()

# Model parameters
full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", lgb.LGBMClassifier(class_weight='balanced')),
])










import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split

# Set seed
np.random.seed(40)

# Initialize
max_score = 0.0
initial_n_estimators = 50
best_params = {'n_estimators':initial_n_estimators, 'learning_rate':0.1, 'max_depth':6, 'subsample':1.00, 'colsample_bytree':1.0, 'reg_lambda':1}

def get_score(params):
    model = Pipeline([
        ("features", features_pipeline),
        ("clf", lgb.LGBMClassifier(class_weight='balanced', **params)),
    ])
    print("Fitting model")
    print(params)
    model.fit(train, train_y)
    prob_y = model.predict_proba(dev)[:, 1]
    precision, recall, _ = precision_recall_curve(dev_y, prob_y, pos_label=1)
    score = max([y for (x,y) in zip(precision, recall) if x >= 0.20])
    return score

# 1) Tune max depth
params = best_params.copy()
for max_depth in [1, 2, 4, 6, 8]:
    params['max_depth'] = max_depth
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned max depth and have a new max score: %.3f" % score)

# 2) Tune subsample
params = best_params.copy()
for subsample in [0.40, 0.60, 0.80, 0.90]:
    params["subsample"] = subsample
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned subsample and have a new max score: %.3f" % score)

# 3) Tune n_estimators
params = best_params.copy()
for n_estimators in [1.1, 1.3]:
    params["n_estimators"] = int(initial_n_estimators*n_estimators)
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned n_estimators and have a new max score: %.3f" % score)

# 4) Tune learning rate
params = best_params.copy()
for n_estimators, learning_rate in zip([initial_n_estimators*1.4, int(initial_n_estimators/1.4)], [0.07, 0.15]):
    params["n_estimators"] = int(n_estimators)
    params["learning_rate"] = learning_rate
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned learning rate and have a new max score: %.3f" % score)

# 5) Tune n_estimators again
params = best_params.copy()
for n_estimators in [int(1.1*initial_n_estimators), int(1.2*initial_n_estimators)]:
    params["n_estimators"] = n_estimators
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned n_estimators again and have a new max score: %.3f" % score)

# 6) Tune sampling by tree
params = best_params.copy()
for colsample_bytree in [0.6, 0.8, 0.9]:
    params["colsample_bytree"] = colsample_bytree
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned colsample_bytree and have a new max score: %.3f" % score)

# 7) Tune subsample again
params = best_params.copy()
for subsample in [0.6, 0.75, 0.9]:
    params["subsample"] = subsample
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned subsample again and have a new max score: %.3f" % score)

# 9) Tune sampling fields
params = best_params.copy()
subsample_ = 0.9 if best_params["subsample"] == 1.0 else best_params["subsample"]
colsample_bytree_ = 0.6 if best_params["colsample_bytree"] == 0.5 else best_params["colsample_bytree"]
for subsample in [subsample_, subsample_ + 0.1]:
    for colsample_bytree in [colsample_bytree_ - 0.1, colsample_bytree_]:
        params["subsample"] = subsample
        params["colsample_bytree"] = colsample_bytree
        score = get_score(params)
        if score > max_score:
            max_score = score
            best_params = params.copy()
            print("Tuned sampling fields and have a new max score: %.3f" % score)

# 10) Tune alpha
params = best_params.copy()
for reg_lambda in [3, 10, 33, 100, 300]:
    params["reg_lambda"] = reg_lambda
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned alpha and have a new max score: %.3f" % score)

# 11) Tune trees
params = best_params.copy()
up = 1
for i in range(5):
    params["n_estimators"] = int(params["n_estimators"] * (1.4 - 0.05*i) if up == 1 else params["n_estimators"] / (1.4 - 0.05*i))
    score = get_score(params)
    if score > max_score:
        max_score = score
        best_params = params.copy()
        print("Tuned n_estimators and have a new max score: %.3f" % score)
    else:
        up = 0 if up == 1 else 1

print("Max score: %.3f" % max_score)
print("Best params: %s" % best_params)
full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", lgb.LGBMClassifier(class_weight='balanced', **best_params)),
])

full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", lgb.LGBMClassifier(class_weight='balanced')),
])

# Tune features pipeline parameters
parameters = {
    'features__num_pipeline__new_numeric_attribs_adder__add_age_booleans': [True, False],
    'features__hybrid_pipeline__new_hybrid_attribs_adder__add_income': [True, False],
    'features__cat_pipeline__drop_unimportant_category_values__drop': [True, False],
    'features__cat_pipeline__cat_encoder__drop': ['first', None]
}

gs = GridSearchCV(full_pipeline, parameters, cv=5)
gs.fit(train, train_y)

print("Best params: %s" % gs.best_params_)
full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", lgb.LGBMClassifier(class_weight='balanced', **best_params, **gs.best_params_)),
])






# Fit
full_pipeline.fit(train, train_y)

# Predict
precision_threshold = 0.20
prob_y = full_pipeline.predict_proba(test)[:, 1]
precision, recall, thresholds = precision_recall_curve(test_y, prob_y, pos_label=1)
print(precision, recall, thresholds)
score = max([y for (x,y) in zip(precision, recall) if x >= precision_threshold])
print('Recall score: %.3f' % score)

# Error analysis
if SHOW_ERROR_ANALYSIS:
    precision_threshold_index = min([i for (x,i) in zip(precision, range(len(precision))) if x >= precision_threshold])
    test["prob_y"] = prob_y
    prob_y_threshold = (list(thresholds) + [1.1])[precision_threshold_index]
    pred_y = (prob_y >= prob_y_threshold).astype(bool)
    print("Prob y Threshold: %.1f" % (prob_y_threshold*100))
    print(confusion_matrix(test_y, pred_y))
    print("Recall: %.1f" % (recall_score(test_y, pred_y)*100))
    print("Precision: %.1f" % (precision_score(test_y, pred_y)*100))
