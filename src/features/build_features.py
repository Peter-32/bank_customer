from numpy import c_, searchsorted
from sklearn.cluster import KMeans
from featexp import univariate_plotter
from pandas import DataFrame, get_dummies
from sklearn.preprocessing import KernelCenterer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
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

def changes_to_dataset(df):
    # Drop highly columns highly correlated with `nr.employed`
    df.drop(["euribor3m", "emp.var.rate"], axis=1, inplace=True)

    # Add/remove features
    df['education_level'] = df['education'].apply(_get_education_level)
    df['expected_income'] = df[['job', 'age']].apply(lambda x: _get_expected_income(*x), axis=1)
    df['expected_bank_balance_by_income'] = df['expected_income'].apply(_get_expected_bank_balance_by_income)
    df['expected_bank_balance_by_age'] = df['age'].apply(_get_expected_bank_balance_by_age)
    df['age_gte_61'] = df['age'].apply(lambda x: 1 if x >= 61 else 0)
    df['age_lte_23'] = df['age'].apply(lambda x: 1 if x <= 23 else 0)
    df.drop(['age', 'expected_bank_balance_by_income'], axis=1, inplace=True)
    df = pd.concat([df.select_dtypes(exclude="O"), \
           pd.get_dummies(df.select_dtypes(include="O"), drop_first=True)], axis=1)
    df["loan_unknown_and_housing_unknown"] = \
        df[["loan_unknown", "housing_unknown"]].\
            apply(lambda x: _get_loan_unknown_and_housing_unknown(*x), axis=1)
    df.drop(["education_university.degree", "loan_unknown", "housing_unknown"], axis=1, inplace=True)


def data_preparation():
    # Extract
    train = read_csv("../../data/interim/train.csv")
    test = read_csv("../../data/interim/test.csv")

    # Prepare the data
    changes_to_dataset(train)
    changes_to_dataset(test)

    # Save the data
    train.to_csv("../../data/interim/prepared_train.csv")
    test.to_csv("../../data/interim/prepared_test.csv")
