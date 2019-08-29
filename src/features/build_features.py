import numpy as np
from pandas import read_csv
from numpy.random import seed
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                                  OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")

def _get_education_level(education):
    education_level = 0
    if education == "illiterate":
        education_level = 0
    elif education == "basic.4y":
        education_level = 1
    elif education == "basic.6y":
        education_level = 2
    elif education == "basic.9y":
        education_level = 3
    elif education == "high.school":
        education_level = 4
    elif education == "professional.course":
        education_level = 6
    elif education == "university.degree":
        education_level = 8
    else:
        education_level = 4
    return education_level

def _get_expected_income(job, age):
    beginner_income = 0
    expert_income = 0
    income = 0
    if job == 'admin.':
        beginner_income = 32500
        expert_income = 64000
    elif job == 'blue-collar':
        beginner_income = 22000
        expert_income = 50000
    elif job == 'technician':
        beginner_income = 25000
        expert_income = 55000
    elif job == 'services':
        beginner_income = 34000
        expert_income = 55000

    elif job == 'management':
        beginner_income = 45000
        expert_income = 122000
    elif job == 'retired':
        beginner_income = 10000
        expert_income = 10000
    elif job == 'entrepreneur':
        beginner_income = 25000
        expert_income = 50000
    elif job == 'self-employed':
        beginner_income = 25000
        expert_income = 50000
    elif job == 'housemaid':
        beginner_income = 19000
        expert_income = 30000
    elif job == 'unemployed':
        beginner_income = 0
        expert_income = 0
    elif job == 'student':
        beginner_income = 0
        expert_income = 0
    elif job == 'unknown':
        beginner_income = 25000
        expert_income = 35000

    income_range = expert_income - beginner_income

    if age <= 19:
        income = beginner_income
    elif age <= 24:
        income = beginner_income + 0.22 * income_range
    elif age <= 34:
        income = beginner_income + 0.65 * income_range
    elif age <= 44:
        income = beginner_income + 0.97 * income_range
    elif age <= 54:
        income = beginner_income + 1.00 * income_range
    elif age <= 64:
        income = beginner_income + 0.99 * income_range
    else:
        income = beginner_income + 0.88 * income_range
    return income

def _get_loan_unknown_and_housing_unknown(loan_unknown, housing_unknown):
    return 1 if loan_unknown == 1 and housing_unknown == 1 else 0

# Not used in this project
def _run_feature_selection(train):
    import shap
    import pandas as pd
    from numpy import cumsum
    from xgboost import XGBClassifier

    seed(40)

    train.fillna(0, inplace=True)

    # X and y
    X = train.drop(["y"], axis=1)
    y = train[["y"]]

    # lightgbm for large number of columns
    # import lightgbm as lgb; clf = lgb.LGBMClassifier()

    # Fit xgboost
    clf = XGBClassifier()
    clf.fit(X, y)

    # shap values
    shap_values = shap.TreeExplainer(clf).shap_values(X[0:10000])

    sorted_feature_importance = pd.DataFrame(shap_values, columns=X.columns).abs().sum().sort_values(ascending=False)
    cumulative_sum = cumsum([y for (x,y) in sorted_feature_importance.reset_index().values])
    gt_999_importance = cumulative_sum / cumulative_sum[-1] > .999
    nth_feature = min([y for (x,y) in zip(gt_999_importance, zip(range(len(gt_999_importance)))) if x])[0]
    important_columns = sorted_feature_importance.iloc[0:nth_feature+1].index.values.tolist()
    important_columns
    return important_columns


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

a_age_ix, a_emp_var_rate_ix, a_euribor3m_ix = 0, 4, 7


class NewNumericAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_age_booleans=True):
        self.add_age_booleans = add_age_booleans
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.add_age_booleans:
            age_gte_61 = np.vectorize(lambda x: 1 if x >= 61 else 0)(X[:, a_age_ix])
            age_lte_23 = np.vectorize(lambda x: 1 if x <= 23 else 0)(X[:, a_age_ix])
            return np.c_[X, age_gte_61, age_lte_23]
        else:
            return np.c_[X]


class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, drop_indicators, drop_age):
        self.drop_indicators = drop_indicators
        self.drop_age = drop_age
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.drop_indicators and self.drop_age:
            new_X = np.delete(X, [a_age_ix, a_emp_var_rate_ix, a_euribor3m_ix], 1)
        elif self.drop_indicators:
            new_X = np.delete(X, [a_emp_var_rate_ix, a_euribor3m_ix], 1)
        elif self.drop_age:
            new_X = np.delete(X, [a_age_ix], 1)
        return np.c[new_X]


b_age_ix, b_job_ix, b_education_ix = 0, 1, 2

class NewHybridAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_income=True):
        self.add_income = add_income
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        education_level = np.vectorize(_get_education_level)(X[:, b_education_ix])
        if self.add_income:
            expected_income = np.vectorize(_get_expected_income)(X[:, b_job_ix], X[:, b_age_ix])
            return np.c_[education_level, expected_income]
        else:
            return np.c_[education_level]


c_job_ix, c_education_ix, c_default_ix = 0, 2, 3
c_loan_ix, c_month_ix, c_poutcome_ix = 5, 7, 9

class DropUnimportantCategoryValues(BaseEstimator, TransformerMixin):
    def __init__(self, drop=True):
        self.drop = drop
    def fit(self, X, y=None):
        self.first_value = [""]*10
        for ix in [c_job_ix, c_education_ix, c_default_ix, c_loan_ix,
                    c_month_ix, c_poutcome_ix]:
            self.first_value[ix] = X[0, ix]
        return self
    def transform(self, X, y=None):
        if not self.drop:
            return np.c_[X]
        else:
            jobs_to_ignore = ["self-employed", "technician", "unemployed",
                              "retired", "entrepreneur", "services",
                              "management"]
            education_to_ignore = ["unknown", "professional.course",
                                   "illiterate"]
            job = np.vectorize(lambda x: x if x not in jobs_to_ignore
                    else self.first_value[c_job_ix])(X[:, c_job_ix])
            education = np.vectorize(lambda x: x if x not in education_to_ignore \
                    else self.first_value[c_education_ix])(X[:, c_education_ix])
            education = np.vectorize(lambda x: x if x not in education_to_ignore \
                    else self.first_value[c_education_ix])(X[:, c_education_ix])
            default = np.vectorize(lambda x: x if x != "yes" \
                    else self.first_value[c_default_ix])(X[:, c_default_ix])
            loan = np.vectorize(lambda x: x if x != "yes" \
                    else self.first_value[c_loan_ix])(X[:, c_loan_ix])
            month = np.vectorize(lambda x: x if x != "sep" \
                    else self.first_value[c_month_ix])(X[:, c_month_ix])
            poutcome = np.vectorize(lambda x: x if x != "nonexistent" \
                    else self.first_value[c_poutcome_ix])(X[:, c_poutcome_ix])
            return np.c_[job, X[:, 1], education, default, X[:, 4],
                         loan, X[:, 6], month, X[:, 8], poutcome]

class LogXShape(BaseEstimator, TransformerMixin):
    def __init__(self, drop=True):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        print(X.shape)
        return X


def data_preparation():
    # Extract
    train = read_csv("../../data/interim/train.csv", nrows=250)

    num_attribs = train.select_dtypes(exclude="O").drop(["y"], axis=1).columns
    hybrid_attribs = train[["age", "job", "education"]].columns
    cat_attribs = train.select_dtypes(include="O").columns

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('new_numeric_attribs_adder', NewNumericAttributesAdder()),
        ('std_scaler', StandardScaler()),
        ('minmax_scaler', MinMaxScaler()),
    ])

    hybrid_pipeline = Pipeline([
        ('selector', DataFrameSelector(hybrid_attribs)),
        ('new_hybrid_attribs_adder', NewHybridAttributesAdder()),
        ('std_scaler', StandardScaler()),
        ('minmax_scaler', MinMaxScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('drop_unimportant_category_values', DropUnimportantCategoryValues()),
        ('cat_encoder', OneHotEncoder(drop='first')),
        # ('log_X_shape', LogXShape()),
    ])

    features_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("hybrid_pipeline", hybrid_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    return features_pipeline
