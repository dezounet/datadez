from __future__ import unicode_literals, print_function

import operator

import numpy as np

from datadez.columns import get_multi_label_occurrence


def multilabel_intersection_matrix(df, column_name):
    """
    Read a multilabel column, output the label intersection matrix:
    For every pair of label, we compute the number of sample where
    these two are both present.

    :param df: input dataframe
    :param column_name: colmun to look for (should contain an iterable)
    :return: np.array of shape (label count, label count)
    """
    # First, get labels occurrence
    occurrences, _ = get_multi_label_occurrence(df[column_name])

    labels = sorted(occurrences.items(), key=operator.itemgetter(1), reverse=True)
    label_to_id = {label[0]: i for i, label in enumerate(labels)}
    labels = [label[0] for label in labels]

    intersection_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for sample in df[column_name]:
        for label_1 in sample:
            label_1_id = label_to_id[label_1]
            for label_2 in sample:
                label_2_id = label_to_id[label_2]

                intersection_matrix[label_1_id][label_2_id] += 1

    return labels, intersection_matrix
