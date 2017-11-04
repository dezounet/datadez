from __future__ import unicode_literals, print_function

import unittest

import numpy as np
import pandas as pd

from plotly.offline import plot

from datadez.dataviz import multilabel_plot

from tests import file_path
from tests.faker import get_random_dataframe


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'mono-label': ['A', 'A', 'B', np.nan, 'C'],
            'multi-label': [['A'], ['A', 'B'], ['B'], [], ['A', 'C', 'D']],
        })

    def test_intersection_matrix(self):
        df = get_random_dataframe(300)
        figure = multilabel_plot.intersection_matrix(df, 'C')

        plot(figure, filename=file_path('chord-diagram.html'))


if __name__ == "__main__":
    unittest.main()
