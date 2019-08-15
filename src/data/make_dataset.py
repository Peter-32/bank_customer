# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
from pathlib import Path
from pprint import pprint
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

def split_data(df):
    # Set seed - ensures that the datasets are split the same way if re-run
    np.random.seed(40)

    # Split datasets
    train, test = train_test_split(df, test_size=0.2)
    _, sample_train = train_test_split(df, test_size=0.2)

    # Diagnostic print statements
    pprint(["DataFrame shapes:", train.shape, test.shape, sample_train.shape, df.shape])
    pprint(["Class Balance per DataFrame:", \
    list(df.y.value_counts(normalize=True).values), \
    list(test.y.value_counts(normalize=True).values), \
    list(train.y.value_counts(normalize=True).values), \
    list(sample_train.y.value_counts(normalize=True).values)])

    # Return datasets
    return train, sample_train, test



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
