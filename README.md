![datadez.png](./docs/datadez.png)

### Pandas dataframe inspection, filtering, balancing [![Build Status][travis-badge]][travis-link] [![MIT License][license-badge]](LICENSE)

The main goal of this package is to make your life easier if you want to:

- Inspect a dataset and compute metrics about its columns content (auto type inference: numeric, mono-label or multi-label).
- Filter the dataset one some criteria (minimum label occurrence, empty example).
- Balance the dataset (TODO) in order to get better performance while training ML or NN models.

### Requirements

- Python 2.7
- Numpy and Pandas

### Do some tests

Just clone this repository, and execute:

    python -m tests.main
    
This will execute a test sample, for you to get what's going on.

[travis-badge]:    https://travis-ci.org/dezounet/datadez.svg?branch=master
[travis-link]:     https://travis-ci.org/dezounet/datadez
[license-badge]:   https://img.shields.io/badge/license-MIT-007EC7.svg
