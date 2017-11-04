from __future__ import unicode_literals, print_function

# Python 2 and 3 compatibility
from builtins import dict

import nltk

import pandas as pd

from datadez.columns import detect_column_type
from datadez.columns import NUMERIC_TYPE, MONO_LABEL_TYPE, MULTI_LABEL_TYPE, TEXT_TYPE

from datadez.vectorize import vectorize_text
from datadez.vectorize import vectorize_mono_label
from datadez.vectorize import vectorize_multi_label

COLUMN_VECTORIZER = {
    NUMERIC_TYPE: None,
    MONO_LABEL_TYPE: vectorize_mono_label,
    MULTI_LABEL_TYPE: vectorize_multi_label,
    TEXT_TYPE: vectorize_text
}


def tokenize(dataset, column):
    """
    Text column to list of words column (multi-label like).

    :param dataset: dataset where data are stored
    :param column: which column should be tokenized

    :return: modified dataset
    """
    dataset[column] = dataset[column].apply(lambda row: nltk.word_tokenize(row[column]), axis=1)

    return dataset


def vectorize_dataset(dataset):
    """
    Fully vectorize a dataset (text, mono-label and multi-label columns).

    :param dataset: dataset to vectorize

    :return: vectorized dataset, vectorizers
    """
    series = {}
    vectorizers = {}

    # Vectorize on column at a time
    for column in dataset.columns:
        column_type = detect_column_type(dataset[column])

        vectorizer_fn = COLUMN_VECTORIZER[column_type]
        if vectorizer_fn is not None:
            vectorized_columns, vectorizer = vectorizer_fn(dataset[column])

            series[column] = vectorized_columns
            vectorizers[column] = vectorizer
        else:
            series[column] = dataset[column].to_frame()
            vectorizers[column] = None

    # Put everything back together
    output_dataset = None
    for column in series.keys():
        vectorized_columns = series[column]
        vectorizer = vectorizers[column]

        if vectorizer is not None:
            sub_index = vectorized_columns.columns
        else:
            sub_index = ["value"]

        # Add one level of index
        vectorized_columns.columns = pd.MultiIndex.from_product([[column], sub_index])

        if output_dataset is not None:
            output_dataset = output_dataset.join(vectorized_columns)
        else:
            output_dataset = vectorized_columns

    return output_dataset, vectorizers
