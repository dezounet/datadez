from __future__ import unicode_literals, print_function

import operator

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer


def _vectorize(vectorizer, series):
    vectorizer.fit(series)

    # Vectorize the input
    vector = vectorizer.transform(series)

    try:
        vector = vector.todense()
        vector = list(np.squeeze(vector))
        vector = list(map(lambda x: np.array(x)[0], vector))
    except AttributeError:
        pass

    # Get vocabulary, ordered by id
    if hasattr(vectorizer, 'vocabulary_'):
        vocabulary = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
        vocabulary = [word[0] for word in vocabulary]
    elif hasattr(vectorizer, 'classes_'):
        vocabulary = vectorizer.classes_
    else:
        raise ValueError("Wrong type of vectorizer given! Excepting one with attribute 'vocabulary_' or 'classes_'")

    # Encapsulate new columns inside a meta column, and put each word to its own column
    new_columns = pd.DataFrame(vector)
    new_columns.columns = pd.Series(vocabulary)

    return new_columns


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
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, binary=binary)
    dataframe = _vectorize(vectorizer, series)

    return dataframe, vectorizer


def vectorize_mono_label(series):
    """
    Vectorize a mono-label column.

    :param series: series to vectorize
    :return: vectorized series as a dataframe, vectorizer
    """
    vectorizer = LabelBinarizer()
    dataframe = _vectorize(vectorizer, series)

    return dataframe, vectorizer


def vectorize_multi_label(series):
    """
    Vectorize a multi-label column.

    :param series: series to vectorize
    :return: vectorized series as a dataframe, vectorizer
    """
    vectorizer = MultiLabelBinarizer()
    dataframe = _vectorize(vectorizer, series)

    return dataframe, vectorizer
