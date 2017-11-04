from __future__ import unicode_literals, print_function

import unittest

import pandas as pd

from datadez.transform import vectorize_dataset


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'text': ["aa bb", "aa", "bb cc", "cc", ""],
            'mono-label': ['A', 'A', 'C', 'B', 'C'],
            'multi-label': [['A'], ['A', 'B'], ['B'], [], ['A', 'C', 'D']],
        })

    def test_mono_label_vectorization(self):
        del self.df['text']
        del self.df['multi-label']

        df, vectorizers = vectorize_dataset(self.df)
        self.assertListEqual(df['mono-label']['A'].tolist(), [1, 1, 0, 0, 0])
        self.assertListEqual(df['mono-label']['B'].tolist(), [0, 0, 0, 1, 0])
        self.assertListEqual(df['mono-label']['C'].tolist(), [0, 0, 1, 0, 1])

    def test_multi_label_vectorization(self):
        del self.df['text']
        del self.df['mono-label']

        df, vectorizers = vectorize_dataset(self.df)
        self.assertListEqual(df['multi-label']['A'].tolist(), [1, 1, 0, 0, 1])
        self.assertListEqual(df['multi-label']['B'].tolist(), [0, 1, 1, 0, 0])
        self.assertListEqual(df['multi-label']['C'].tolist(), [0, 0, 0, 0, 1])
        self.assertListEqual(df['multi-label']['D'].tolist(), [0, 0, 0, 0, 1])

    def test_text_vectorization(self):
        del self.df['mono-label']
        del self.df['multi-label']

        df, vectorizers = vectorize_dataset(self.df)
        self.assertListEqual(df['text']['aa'].tolist(), [1, 1, 0, 0, 0])
        self.assertListEqual(df['text']['bb'].tolist(), [1, 0, 1, 0, 0])
        self.assertListEqual(df['text']['cc'].tolist(), [0, 0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
