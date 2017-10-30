from __future__ import unicode_literals, print_function

import pprint

import pandas as pd

from datadez.filter import filter_empty
from datadez.filter import filter_small_occurrence
from datadez.summarize import summarize
from tests.faker import get_random_dataframe

if __name__ == "__main__":
    # Get nice output
    pd.set_option('display.width', 150)

    df = get_random_dataframe(100)
    print("Starting from this dataframe (len=%s):" % len(df))
    print(df.head(), "\n")

    print("With these metrics:")
    df_summaries = summarize(df)
    pprint.pprint(df_summaries)

    print("\nFiltering the small occurrences label of columns B and C...")
    df = filter_small_occurrence(df, 'B', 3)
    df = filter_small_occurrence(df, 'C', 6)

    print("We now get this (len=%s):" % len(df))
    print(df.head(), "\n")

    print("\nFiltering empty entry example for column B or C...")
    df = filter_empty(df, ['B', 'C'])

    print("\nWe finally have a clean dataframe (len=%s):" % len(df))
    print(df.head(), "\n")

    print("With these metrics:")
    df_summaries = summarize(df)
    pprint.pprint(df_summaries)
