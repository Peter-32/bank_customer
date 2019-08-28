import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                                  OneHotEncoder
from pandas import read_csv
import warnings
warnings.filterwarnings("ignore")

def _get_expected_bank_balance_by_age(age):
    balance = 0
    if age < 35:
        balance = 4000
    elif age <= 44:
        balance = 6000
    elif age <= 54:
        balance = 9000
    elif age <= 64:
        balance = 10000
    elif age <= 74:
        balance = 16000
    else:
        balance = 12000
    return balance

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


def _get_expected_bank_balance_by_income(income):
    balance = 0
    if income < 25000:
        balance = 2500
    elif income < 45000:
        balance = 3500
    elif income < 70000:
        balance = 5000
    elif income < 115000:
        balance = 8000
    else:
        balance = 12000
    return balance

def _get_loan_unknown_and_housing_unknown(loan_unknown, housing_unknown):
    return 1 if loan_unknown == 1 and housing_unknown == 1 else 0

from sklearn.base import BaseEstimator, TransformerMixin

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
        expected_bank_balance_by_age = np.vectorize(_get_expected_bank_balance_by_age)(X[:, a_age_ix])
        if self.add_age_booleans:
            age_gte_61 = np.vectorize(lambda x: 1 if x >= 61 else 0)(X[:, a_age_ix])
            age_lte_23 = np.vectorize(lambda x: 1 if x <= 23 else 0)(X[:, a_age_ix])
            return np.c_[X, expected_bank_balance_by_age, age_gte_61, age_lte_23]
        else:
            return np.c_[X, expected_bank_balance_by_age]

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
            expected_bank_balance_by_income = np.vectorize(_get_expected_bank_balance_by_income)(expected_income)
            return np.c_[education_level, expected_income, expected_bank_balance_by_income]
        else:
            return np.c_[education_level]

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
        ('cat_encoder', OneHotEncoder(drop='first')),
    ])

    from sklearn.pipeline import FeatureUnion

    features_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("hybrid_pipeline", hybrid_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    return features_pipeline
