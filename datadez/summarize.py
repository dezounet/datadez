from __future__ import unicode_literals, print_function

from datadez.columns import NUMERIC_TYPE, MONO_LABEL_TYPE, MULTI_LABEL_TYPE
from datadez.columns import detect_column_type
from datadez.summary import mono_label_summary
from datadez.summary import multi_label_summary
from datadez.summary import numeric_summary

COLUMN_TYPE_SUMMARIZER = {
    NUMERIC_TYPE: numeric_summary,
    MONO_LABEL_TYPE: mono_label_summary,
    MULTI_LABEL_TYPE: multi_label_summary,
}


def summarize(dataset):
    summaries = {}

    for column in dataset:
        column_type = detect_column_type(dataset[column])

        if column_type in COLUMN_TYPE_SUMMARIZER.keys():
            summary = COLUMN_TYPE_SUMMARIZER[column_type](dataset[column])
        else:
            summary = None

        summaries[column] = summary

    return summaries
