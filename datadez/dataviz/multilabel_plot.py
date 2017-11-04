from __future__ import unicode_literals, print_function

from datadez.multilabel import multilabel_intersection_matrix
from datadez.dataviz.chord_diagram import chord_diagram


def intersection_matrix(dataset, column):
    """
    Compute and plot intersection matrix for a multilabel column.
    """
    labels, matrix = multilabel_intersection_matrix(dataset, column)
    figure = chord_diagram(matrix, labels)

    return figure
