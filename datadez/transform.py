from __future__ import unicode_literals, print_function

import numpy as np

import nltk

from sklearn.feature_extraction.text import CountVectorizer


def tokenize(dataset, column):
    """
    Text column to list of words column.

    :param dataset: dataset where data are stored
    :param column: which column should be tokenized

    :return: modified dataset
    """
    dataset[column] = dataset[column].apply(lambda row: nltk.word_tokenize(row[column]), axis=1)

    return dataset


def vectorize_text(dataset, column, min_df=1, max_df=1.0, binary=False):
    """
    Vectorize a column, put it back to the dataframe.

    :param dataset: dataset where data are stored
    :param column: which column should be vectorized
    :param min_df: float in range [0.0, 1.0] or int, default=1
    :param max_df: float in range [0.0, 1.0] or int, default=1.0
    :param binary: If True, all non zero counts are set to 1, else to count.

    :return: modified dataset
    """
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, binary=binary)
    vectorizer.fit(dataset[column])

    dataset[column] = list(np.squeeze(vectorizer.transform(dataset[column]).todense()))
    dataset[column] = dataset[column].apply(lambda x: np.array(x)[0])

    # Encapsulate new columns inside a meta column
    new_columns = pd.DataFrame(dataset[column].tolist())
    new_columns.columns = pd.MultiIndex.from_product([[column], new_columns.columns])

    # TODO: instead of index, column name should be word

    # Remove old column
    del dataset[column]

    # Adapt original level
    dataset.columns = pd.MultiIndex.from_product([dataset.columns, ['dummy']])

    # Append new columns to the original dataframe
    dataset = dataset.join(new_columns)

    return dataset, vectorizer


def text_to_multi_label(dataset, column):
    # TODO
    return dataset


def multi_label_to_one_hot(dataset, column):
    # TODO
    return dataset


if __name__ == "__main__":
    import pandas as pd

    from tests.faker import get_random_dataframe

    # Get nice output
    pd.set_option('display.width', 250)

    df = get_random_dataframe(100)
    print("Original dataframe:")
    print(df.head())

    df, vectorizer = vectorize_text(df, 'D')

    print("Vectorized dataframe with a new level:")
    print(df.head())
