from __future__ import unicode_literals, print_function

# Python 2 and 3 compatibility
from builtins import dict

import numpy as np

from datadez.columns import MONO_LABEL_TYPE
from datadez.columns import MULTI_LABEL_TYPE
from datadez.columns import NUMERIC_TYPE
from datadez.columns import get_mono_label_occurrence
from datadez.columns import get_multi_label_occurrence


def numeric_summary(column):
    mean = np.mean(column)
    std = np.std(column)

    return {
        'column_type': NUMERIC_TYPE,
        'mean': mean,
        'std': std,
    }


def mono_label_summary(column):
    label_occurrences = get_mono_label_occurrence(column).to_dict()
    occurrences = [v for v in label_occurrences.values()]
    max_count = max(occurrences)
    min_count = min(occurrences)
    mean_count = np.mean(occurrences)
    std_count = np.std(occurrences)

    return {
        'column_type': MONO_LABEL_TYPE,
        'labels': len(label_occurrences),
        'occurrence_max': max_count,
        'occurrence_min': min_count,
        'occurrence_mean': mean_count,
        'occurrence_std_dev': std_count,
        'imbalance_ratio': max_count / min_count,
    }


def multi_label_summary(column):
    occurrences, cardinalities = get_multi_label_occurrence(column)
    occurrences = [v for v in occurrences.to_dict().values()]
    max_count = max(occurrences)
    min_count = min(occurrences)
    mean_count = np.mean(occurrences)
    std_count = np.std(occurrences)

    mean_cardinality = np.mean(cardinalities)
    std_cardinality = np.std(cardinalities)

    # Get some stats on label grouping, considering the column
    # as a mono-label column
    subset_summary = mono_label_summary(column.astype(str))
    del (subset_summary['column_type'])

    return {
        'column_type': MULTI_LABEL_TYPE,
        'labels': len(occurrences),
        'occurrence_max': max_count,
        'occurrence_min': min_count,
        'occurrence_mean': mean_count,
        'occurrence_std_dev': std_count,
        'imbalance_ratio': max_count / min_count,
        'cardinality_mean': mean_cardinality,
        'cardinality_std_dev': std_cardinality,
        'partitions': subset_summary
    }
