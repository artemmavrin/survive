=============
API Reference
=============

This page lists all the public classes and functions of Survive.

:mod:`survive`: Top-Level Module
=================================

.. automodule:: survive
    :no-members:
    :no-inherited-members:

Classes
-------

.. currentmodule:: survive

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    SurvivalData


:mod:`survive.base`: Base Classes
=================================

.. automodule:: survive.base
    :no-members:
    :no-inherited-members:

Classes
-------

.. currentmodule:: survive

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    base.Fittable
    base.Model
    base.Predictor
    base.Summary


:mod:`survive.datasets`: Survival Datasets
==========================================

.. automodule:: survive.datasets
    :no-members:
    :no-inherited-members:

Functions
---------

.. currentmodule:: survive

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    datasets.channing
    datasets.leukemia


:mod:`survive.nonparametric`: Nonparametric Estimation
=======================================================

.. automodule:: survive.nonparametric
    :no-members:
    :no-inherited-members:

Estimator Classes
-----------------

.. currentmodule:: survive

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    nonparametric.KaplanMeier
    nonparametric.NelsonAalen

Summary Classes
---------------

.. currentmodule:: survive

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    nonparametric.NonparametricEstimatorSummary

Abstract Base Classes
---------------------

.. currentmodule:: survive

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    nonparametric.NonparametricEstimator
    nonparametric.NonparametricSurvival
