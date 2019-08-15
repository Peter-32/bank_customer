# Imports
import matplotlib.pyplot as plt
from featexp import get_univariate_plots, get_trend_stats

# Options
plt.rcParams['figure.figsize'] = 16, 16

def histograms_for_nonbinary_columns(sample_train):
    # Find boolean columns and ignore them
    bool_cols = [col for col in sample_train if sample_train[col].value_counts().index.isin([0,1]).all()]

    # Make the histogram
    sample_train[[x for x in sample_train.columns if x not in bool_cols]].hist()

def featexp_plots(sample_train):
    # Find boolean columns and ignore them
    bool_cols = [col for col in sample_train if sample_train[col].value_counts().index.isin([0,1]).all()]

    # Get data
    data = sample_train[[x for x in sample_train.columns if x == 'y' or \
                                                            x not in bool_cols]]
                                                            
    # Make plots
    get_univariate_plots(data=data, target_col='y',
                         features_list=data.columns, bins=10)
