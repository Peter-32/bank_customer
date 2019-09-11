# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules
from features.build_features import *

# Public modules
import seaborn as sns
import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, \
                            precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from numpy.random import seed
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
                            f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
                                          QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

# Inputs
SHOW_ERROR_ANALYSIS = True

# Extract
seed(40)
train = read_csv("../../data/interim/train.csv", nrows=7500)
train_y = train[["y"]].values
dev = read_csv("../../data/interim/dev.csv", nrows=7500)
dev_y = dev[["y"]].values

# Data parameters
features_pipeline = data_preparation()

# Model parameters
full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", lgb.LGBMClassifier(class_weight='balanced')),
])

models = []
models.append(('LightGBM', lgb.LGBMClassifier(class_weight='balanced')))
models.append(('LR_lib', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN1', KNeighborsClassifier(n_neighbors=1)))
models.append(('KNN2', KNeighborsClassifier(n_neighbors=2)))
models.append(('SVCPoly2', SVC(kernel='poly', degree=2, probability=True)))
models.append(('SVCRbf2', SVC(kernel='rbf', degree=2, probability=True)))
models.append(('SVCRbf3', SVC(kernel='rbf', degree=3, probability=True)))
models.append(('DT9', DecisionTreeClassifier(max_depth=9)))
models.append(('RF', RandomForestClassifier()))
models.append(('GBR', GradientBoostingClassifier()))

# Removed due to low performance
models.append(('DT1', DecisionTreeClassifier(max_depth=1)))
models.append(('DT2', DecisionTreeClassifier(max_depth=2)))
models.append(('DT3', DecisionTreeClassifier(max_depth=3)))
models.append(('DT4', DecisionTreeClassifier(max_depth=4)))
models.append(('DT5', DecisionTreeClassifier(max_depth=5)))
models.append(('SVCPoly3', SVC(kernel='poly', degree=3, probability=True)))
models.append(('SVCLinear', SVC(kernel='linear', probability=True)))
models.append(('SVCSigmoid', SVC(kernel='sigmoid', probability=True)))

# Removed due to being slow
models.append(('KNN3', KNeighborsClassifier(n_neighbors=3)))
models.append(('KNN5', KNeighborsClassifier(n_neighbors=5)))
models.append(('KNN7', KNeighborsClassifier(n_neighbors=7)))

# Redundant
models.append(('LR', LogisticRegression()))

# Evaluate each model
scores = []
names = []
precision_threshold = 0.20
for name, clf in models:
    full_pipeline = Pipeline([
        ("features", features_pipeline),
        ("clf", clf),
    ])
    full_pipeline.fit(train, train_y)
    prob_y = full_pipeline.predict_proba(dev)[:, 1]
    precision, recall, _ = precision_recall_curve(dev_y, prob_y, pos_label=1)
    score = max([y for (x,y) in zip(precision, recall) if x >= precision_threshold])
    scores.append(score)
    names.append(name)
    msg = "%s: %f" % (name, scores[-1])
    print(msg)

# Put results in dataframe
models_df = DataFrame({'name': names, 'score': scores}).sort_values(by=['score'], ascending=False).iloc[0:]

# Plot
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8
ax = sns.barplot(x="name", y="score", data=models_df)
ax.set_xticklabels(models_df['name'], rotation=75, fontdict={'fontsize': 12})
plt.savefig('../../reports/figures/dev_sca_scores.png')
plt.show()
