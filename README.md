# `survive`

[![PyPI version](https://badge.fury.io/py/survive.svg)](https://badge.fury.io/py/survive)

Survival analysis in Python.

## Installation

The latest version of `survive` can be installed directly after cloning from GitHub:

    git clone https://github.com/artemmavrin/survive.git
    cd survive
    make install

Moreover, `survive` is on the [Python Package Index (PyPI)](https://pypi.org/project/survive/), so a recent version of it can be installed with the `pip` utility:

    pip install survive

## Dependencies

* [NumPy](http://www.numpy.org)
* [SciPy](https://www.scipy.org)
* [pandas](https://pandas.pydata.org)
* [Matplotlib](https://matplotlib.org)

## Case Studies

* [Leukemia remission time dataset.](https://github.com/artemmavrin/survive/blob/master/examples/Leukemia%20Remission%20Time%20Dataset.ipynb)
  A small dataset (42 total observations) separated into two groups with heavy right-censoring in one and none in the other.
* [Channing House dataset.](https://github.com/artemmavrin/survive/blob/master/examples/Channing%20House%20Dataset.ipynb)
  A slightly larger dataset (462 total observations) separated into two groups with right-censoring and left truncation in both.
