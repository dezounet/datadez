from __future__ import unicode_literals, print_function

from past.builtins import basestring

import numbers
from collections import defaultdict

import numpy as np
import pandas as pd

NUMERIC_TYPE = 'numeric'
MONO_LABEL_TYPE = 'mono-label'
MULTI_LABEL_TYPE = 'multi-label'
TEXT_TYPE = 'text'


def get_mono_label_occurrence(column):
    return column.value_counts(dropna=False)


def get_multi_label_occurrence(column):
    counter = defaultdict(int)
    cardinalities = []

    for labels in column:
        cardinalities.append(len(labels))
        for label in labels:
            counter[label] += 1

    counter = pd.Series(counter)

    return counter, cardinalities


def detect_column_type(column):
    assert len(column) > 0

    current_entry = 0
    column_type = None
    while column_type is None and current_entry < len(column):
        entry = column.iloc[current_entry]

        if isinstance(entry, basestring) and " " in entry:
            column_type = TEXT_TYPE
        elif isinstance(entry, basestring):
            column_type = MONO_LABEL_TYPE
        elif isinstance(entry, (list, set, tuple)):
            column_type = MULTI_LABEL_TYPE
        elif not np.isnan(entry) and isinstance(entry, numbers.Number):
            column_type = NUMERIC_TYPE

        current_entry += 1

    return column_type
