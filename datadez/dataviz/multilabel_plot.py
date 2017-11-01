from __future__ import unicode_literals, print_function

from datadez.multilabel import multilabel_intersection_matrix
from datadez.dataviz.chord_diagram import plot_chord_diagram


def plot_intersection_matrix(dataset, column):
    """
    Compute and plot intersection matrix for a multilabel column.
    """
    labels, matrix = multilabel_intersection_matrix(dataset, column)
    plot_chord_diagram(matrix, labels)


if __name__ == "__main__":
    from tests.faker import get_random_dataframe

    df = get_random_dataframe(300)
    plot_intersection_matrix(df, 'C')
