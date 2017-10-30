![datadez.png](./docs/datadez.png)

### Pandas dataframe inspection, filtering, balancing [![Build Status][travis-badge]][travis-link] [![MIT License][license-badge]](LICENSE)

The main goal of this package is to make your life easier if you want to:

- Inspect a dataset and compute metrics about its columns content (auto type inference: numeric, mono-label or multi-label).
- Filter the dataset one some criteria (minimum label occurrence, empty example).
- Balance the dataset (TODO) in order to get better performance while training ML or NN models.

### Requirements

- Python 2.7 or 3.6
- Numpy and Pandas

### Usage

```python
# Dataframe with numeric, mono-label, or multi-label (list, tuple, set) columns
df = pd.DataFrame(...)

# Filter label not occurring much in column 'B'
from datadez.filter import filter_small_occurrence
df = datadez.filter.filter_small_occurrence(df, column_name='B', min_occurrence=3)

# Filter empty row based on column 'B' or 'C' values
from datadez.filter import filter_empty
df = filter_empty(df, column_names=['B', 'C'])

# Compute some metrics about your dataset
import pprint
from datadez.summarize import summarize
df_summaries = summarize(df)
pprint.pprint(df_summaries)
```

### Do some tests

Just clone this repository, and execute:

    python -m tests.main
    
This will execute a test sample, for you to get what's going on:

    Starting from this dataframe (len=100):
              A       B                         C                                  D
    0 -1.586145  6MDH0R          [G82DAB, LUK187]       puppy weeps adorable crazily
    1  1.615922  FZPGMF  [53N7FZ, 4PUXVZ, T13QZ8]          donkey hits dirty crazily
    2  0.415307  U63ZTQ                        []        house hits odd occasionally
    3  1.127916  D2LDB2                  [D2LDB2]  rabbit drives stupid occasionally
    4 -0.235374  1ESVPW          [JA6MD0, T13QZ8]         donkey swims odd dutifully
    
    With these metrics:
    {u'A': {u'column_type': u'numeric',
            u'mean': 0.03523764999327153,
            u'std': 0.87669461972996077},
     u'B': {u'column_type': u'mono-label',
            u'imbalance_ratio': 6,
            u'labels': 29,
            u'occurrence_max': 6,
            u'occurrence_mean': 3.4482758620689653,
            u'occurrence_min': 1,
            u'occurrence_std_dev': 1.5444272880385306},
     u'C': {u'cardinality_mean': 2.04,
            u'cardinality_std_dev': 1.2955307792561319,
            u'column_type': u'multi-label',
            u'imbalance_ratio': 4,
            u'labels': 30,
            u'occurrence_max': 12,
            u'occurrence_mean': 6.7999999999999998,
            u'occurrence_min': 3,
            u'occurrence_std_dev': 2.5086516962969037},
     u'D': {u'column_type': u'mono-label',
            u'imbalance_ratio': 2,
            u'labels': 94,
            u'occurrence_max': 2,
            u'occurrence_mean': 1.0638297872340425,
            u'occurrence_min': 1,
            u'occurrence_std_dev': 0.24444947432076719}}
    
    Filtering the small occurrences label of columns B and C...
    We now get this (len=100):
              A       B                 C                                  D
    0 -1.586145  6MDH0R  [G82DAB, LUK187]       puppy weeps adorable crazily
    1  1.615922     NaN  [4PUXVZ, T13QZ8]          donkey hits dirty crazily
    2  0.415307  U63ZTQ                []        house hits odd occasionally
    3  1.127916  D2LDB2          [D2LDB2]  rabbit drives stupid occasionally
    4 -0.235374  1ESVPW  [JA6MD0, T13QZ8]         donkey swims odd dutifully
    
    
    Filtering empty entry example for column B or C...
    
    We finally have a clean dataframe (len=71):
              A       B                         C                                  D
    0 -1.586145  6MDH0R          [G82DAB, LUK187]       puppy weeps adorable crazily
    3  1.127916  D2LDB2                  [D2LDB2]  rabbit drives stupid occasionally
    4 -0.235374  1ESVPW          [JA6MD0, T13QZ8]         donkey swims odd dutifully
    6 -0.372390  XF2L23  [D2LDB2, 471R31, U63ZTQ]         puppy barfs stupid crazily
    9  0.064844  471R31                  [89LMS2]   rabbit drives clueless foolishly
    
    With these metrics:
    {u'A': {u'column_type': u'numeric',
            u'mean': -0.08669925159868834,
            u'std': 0.88961458709106267},
     u'B': {u'column_type': u'mono-label',
            u'imbalance_ratio': 2,
            u'labels': 20,
            u'occurrence_max': 5,
            u'occurrence_mean': 3.5499999999999998,
            u'occurrence_min': 2,
            u'occurrence_std_dev': 1.0712142642814275},
     u'C': {u'cardinality_mean': 1.971830985915493,
            u'cardinality_std_dev': 0.93404814825854943,
            u'column_type': u'multi-label',
            u'imbalance_ratio': 2,
            u'labels': 19,
            u'occurrence_max': 12,
            u'occurrence_mean': 7.3684210526315788,
            u'occurrence_min': 5,
            u'occurrence_std_dev': 2.005532514027121},
     u'D': {u'column_type': u'mono-label',
            u'imbalance_ratio': 2,
            u'labels': 69,
            u'occurrence_max': 2,
            u'occurrence_mean': 1.0289855072463767,
            u'occurrence_min': 1,
            u'occurrence_std_dev': 0.16776575221435114}}

[travis-badge]:    https://travis-ci.org/dezounet/datadez.svg?branch=master
[travis-link]:     https://travis-ci.org/dezounet/datadez
[license-badge]:   https://img.shields.io/badge/license-MIT-007EC7.svg
