#!/usr/bin/env python

from setuptools import setup

setup(
    name="datadez",
    version="0.1.0",
    description="Inspect, filter and balance your dataset",
    keywords=["dataset", "inspect", "filter", "balance"],
    author="dezounet",
    maintainer="dezounet",
    author_email="dezonthenet@gmail.com",
    license="MIT",
    packages=[
        "datadez",
    ],
    url="https://github.com/dezounet/datadez",
    download_url='https://github.com/dezounet/datadez/archive/0.1.tar.gz',
    install_requires=[
        "numpy>=1.13.3",
        "pandas>=0.21.0",
        "future",
    ],
)
