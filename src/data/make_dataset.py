# -*- coding: utf-8 -*-
import click
import logging
from numpy.random import seed
from pandas import read_csv
from pathlib import Path
from pprint import pprint
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def make_train_test_sets():
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
    train, test = train_test_split(df, test_size=0.2)

    # Write results to files
    train.to_csv("../../data/interim/train.csv", index=False)
    test.to_csv("../../data/interim/test.csv", index=False)

def dataset_modifications_after_exploring_data():
    # Extract
    train = read_csv("../../data/interim/train.csv")
    test = read_csv("../../data/interim/test.csv")

    # Drop highly columns highly correlated with `nr.employed`
    train.drop(["euribor3m", "emp.var.rate"], axis=1, inplace=True)
    test.drop(["euribor3m", "emp.var.rate"], axis=1, inplace=True)

    train["campaign"] = train["campaign"].apply(lambda x: x - 1)
    test["campaign"] = test["campaign"].apply(lambda x: x - 1)

    # Write results to files
    train.to_csv("../../data/interim/train_v2.csv", index=False)
    test.to_csv("../../data/interim/test_v2.csv", index=False)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()