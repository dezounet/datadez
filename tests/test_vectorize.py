from __future__ import unicode_literals, print_function

import unittest

import numpy as np
import pandas as pd

from datadez.transform import vectorize_dataset


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'mono-label': ['A', 'A', 'C', 'B', 'C'],
            'multi-label': [['A'], ['A', 'B'], ['B'], [], ['A', 'C', 'D']],
        })

    def test_mono_label_vectorization(self):
        print(self.df.head())

        df = vectorize_dataset(self.df)

        print(df)


if __name__ == "__main__":
    unittest.main()
