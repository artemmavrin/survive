=======
Survive
=======

---------------------------
Survival Analysis in Python
---------------------------

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://badge.fury.io/py/survive
    :alt: Python Version

.. image:: https://badge.fury.io/py/survive.svg
    :target: https://badge.fury.io/py/survive
    :alt: Package Version

.. image:: https://readthedocs.org/projects/survive-python/badge/?version=latest
    :target: https://survive-python.readthedocs.io/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/artemmavrin/survive.svg
    :target: https://github.com/artemmavrin/survive/blob/master/LICENSE
    :alt: GitHub License

Survive is a Python 3 package built on top of `NumPy <http://www.numpy.org>`__
and `pandas <https://pandas.pydata.org>`__ that provides statistical tools for
the analysis of survival, lifetime, and event data.

Website: https://survive-python.readthedocs.io


Installation
~~~~~~~~~~~~

The latest version of Survive can be installed directly after cloning from
`GitHub <https://github.com/artemmavrin/survive>`__.

.. code-block :: shell

  git clone https://github.com/artemmavrin/survive.git
  cd survive
  make install

Moreover, Survive is on the
`Python Package Index (PyPI) <https://pypi.org/project/survive/>`__, so a recent
version of it can be installed with the `pip <https://pip.pypa.io/en/stable/>`__
tool.

.. code-block :: shell

  python -m pip install survive


Dependencies
~~~~~~~~~~~~

Survive relies on the following scientific computing packages.

* `NumPy <http://www.numpy.org>`__
* `pandas <https://pandas.pydata.org>`__
* `SciPy <https://www.scipy.org>`__
* `Matplotlib <https://matplotlib.org>`__


Case Studies
~~~~~~~~~~~~

`Leukemia remission times <https://survive-python.readthedocs.io/examples/Leukemia_Remission_Time_Dataset.html>`__
    A small dataset (42 total observations) separated into two groups with heavy
    right-censoring in one and none in the other.

`Channing House data <https://survive-python.readthedocs.io/examples/Channing_House_Dataset.html>`__
    A slightly larger dataset (462 total observations) separated into two groups
    with right-censoring and left truncation in both.


API Reference
~~~~~~~~~~~~~

The complete API Reference for Survive is available on
`Read the Docs <https://survive-python.readthedocs.io/api.html>`__.
