from __future__ import unicode_literals, print_function

import unittest

from collections import defaultdict

import numpy as np
import pandas as pd

from datadez.filter import filter_small_occurrence
from datadez.filter import filter_empty


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'mono-label': ['A', 'A', 'B', np.nan, 'C'],
            'multi-label': [['A'], ['A', 'B'], ['B'], [], ['A', 'C', 'D']],
        })

    def test_filter_occurences_mono_label(self):
        df = filter_small_occurrence(self.df, 'mono-label', 2)

        # Check rows has not been removed
        self.assertEqual(len(df), len(self.df))

        # Check occurrences are as expected
        occurrences = df['mono-label'].value_counts(dropna=False)
        self.assertEqual(occurrences['A'], 2)
        self.assertEqual(occurrences.loc[occurrences.index.isnull()].values, 3)
        self.assertListEqual(df['mono-label'].tolist(), ['A', 'A', np.nan, np.nan, np.nan])

    def test_filter_occurrences_multi_label(self):
        df = filter_small_occurrence(self.df, 'multi-label', 3)

        # Check rows has not been removed
        self.assertEqual(len(df), len(self.df))

        occurrences = defaultdict(int)
        for entry in df['multi-label']:
            for label in entry:
                occurrences[label] += 1

        # Check occurrences are as expected
        self.assertEqual(len(occurrences), 1)
        self.assertEqual(occurrences['A'], 3)
        self.assertListEqual(df['multi-label'].tolist(), [['A'], ['A'], [], [], ['A']])

    def test_filter_empty_mono_label(self):
        df = filter_empty(self.df, ['mono-label'])

        # Empty row for column 'mono-label' is filtered
        self.assertEqual(len(df), 4)
        self.assertListEqual(df['mono-label'].tolist(), ['A', 'A', 'B', 'C'])

    def test_filter_empty_multi_label(self):
        df = filter_empty(self.df, ['multi-label'])

        # Empty row for column 'multi-label' is filtered
        self.assertEqual(len(df), 4)
        self.assertListEqual(df['multi-label'].tolist(), [['A'], ['A', 'B'], ['B'], ['A', 'C', 'D']])


if __name__ == "__main__":
    unittest.main()
