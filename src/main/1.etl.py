# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules


# Public modules
from numpy.random import seed
from pandas import read_csv
from sklearn.model_selection import train_test_split

# Set seed - ensures that the datasets are split the same way if re-run
seed(40)

# Extract
df = read_csv("../../data/raw/bank-additional-full.csv", sep=";")

# These look like duplicates because it seems unlikely there would
    # be two records with the same duration value.
df.drop_duplicates(inplace=True)

# Transform target to numeric
df['y'] = df['y'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)

# Remove obvious data leakage column
df.drop(["duration"], axis=1, inplace=True)

# Split datasets
temp, test = train_test_split(df, test_size=0.2)
train, dev = train_test_split(temp, test_size=0.25)

# Write results to files
train.to_csv("../../data/interim/train.csv", index=False)
dev.to_csv("../../data/interim/dev.csv", index=False)
test.to_csv("../../data/interim/test.csv", index=False)
