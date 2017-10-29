from __future__ import unicode_literals, print_function

from copy import deepcopy

import numpy as np

from datadez.columns import MONO_LABEL_TYPE, MULTI_LABEL_TYPE
from datadez.columns import detect_column_type
from datadez.columns import get_mono_label_occurrence
from datadez.columns import get_multi_label_occurrence


def _filter_mono_label_small_occurrence(column, min_occurrence):
    occurrences = get_mono_label_occurrence(column)
    labels_to_delete = occurrences[occurrences < min_occurrence].keys()

    mask = column.isin(labels_to_delete)

    # Should check why, with no deepcopy, pandas is throwing a warning...
    filtered_column = deepcopy(column)
    filtered_column.loc[mask] = np.nan

    return filtered_column


def _filter_multi_label_small_occurrence(column, min_occurrence):
    occurrences, _ = get_multi_label_occurrence(column)
    labels_to_delete = occurrences[occurrences < min_occurrence].keys()

    for labels in column:
        labels[:] = [label for label in labels if label not in labels_to_delete]

    filtered_column = column
    return filtered_column


def filter_small_occurrence(column, min_occurrence):
    column_type = detect_column_type(column)

    if column_type == MONO_LABEL_TYPE:
        filtered_column = _filter_mono_label_small_occurrence(column, min_occurrence)
    elif column_type == MULTI_LABEL_TYPE:
        filtered_column = _filter_multi_label_small_occurrence(column, min_occurrence)
    else:
        raise NotImplementedError

    return filtered_column


def _drop_nan(dataset, column_name):
    return dataset.dropna(subset=[column_name])


def _filter_empty_mono_label(dataset, column_name):
    dataset = _drop_nan(dataset, column_name)
    return dataset


def _filter_empty_multi_label(dataset, column_name):
    dataset = _drop_nan(dataset, column_name)
    dataset = dataset[dataset[column_name].apply(len) > 0]
    return dataset


def filter_empty(dataset, column_names):
    for column_name in column_names:
        column = dataset[column_name]
        column_type = detect_column_type(column)

        if column_type == MONO_LABEL_TYPE:
            dataset = _filter_empty_mono_label(dataset, column_name)
        elif column_type == MULTI_LABEL_TYPE:
            dataset = _filter_empty_multi_label(dataset, column_name)
        else:
            raise NotImplementedError

    return dataset
