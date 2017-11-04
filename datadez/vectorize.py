from __future__ import unicode_literals, print_function

import operator

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


def vectorize_text(series, min_df=1, max_df=1.0, binary=False):
    """
    Vectorize a text column.

    Tokenization of the input and vectorization is done
    through a CountVectorizer.

    :param series: series to vectorize
    :param min_df: float in range [0.0, 1.0] or int, default=1
    :param max_df: float in range [0.0, 1.0] or int, default=1.0
    :param binary: If True, all non zero counts are set to 1, else to count.

    :return: vectorized series as a dataframe, vectorizer
    """
    # Fit the vectorizer
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, binary=binary)
    vectorizer.fit(series)

    # Vectorize the text
    vector = list(np.squeeze(vectorizer.transform(series).todense()))
    vector = map(lambda x: np.array(x)[0], vector)

    # Get vocabulary, ordered by id
    vocabulary = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
    vocabulary = [word[0] for word in vocabulary]

    # Encapsulate new columns inside a meta column, and put each word to its own column
    new_columns = pd.DataFrame(vector)
    new_columns.columns = vocabulary

    return new_columns, vectorizer


def vectorize_mono_label(series):
    """
    Vectorize a mono-label column.

    :param series: series to vectorize
    :return: vectorized series as a dataframe, vectorizer
    """
    new_columns, vectorizer = vectorize_text(series, min_df=1, max_df=1.0, binary=True)

    return new_columns, vectorizer


def vectorize_multi_label(series):
    """
    Vectorize a multi-label column.

    :param series: series to vectorize
    :return: vectorized series as a dataframe, vectorizer
    """
    series = series.apply(lambda x: " ".join(x))
    new_columns, vectorizer = vectorize_text(series, min_df=1, max_df=1.0, binary=True)

    return new_columns, vectorizer
