"""Unit tests for the SurvivalData class."""

import pytest

import numpy as np

from survive import SurvivalData
from survive import datasets


def test_init_with_dataframe():
    """Test initializing with a DataFrame."""
    # Leukemia dataset
    leukemia = datasets.leukemia()

    # Initialize with column names
    SurvivalData("time", status="status", group="group", data=leukemia)

    # Initialize with arrays
    SurvivalData(leukemia.time, status=leukemia.status, group=leukemia.group,
                 data=leukemia)

    # Channing House dataset
    channing = datasets.channing()

    # Initialize with column names
    SurvivalData("exit", entry="entry", status="status", group="sex",
                 data=channing, warn=False)

    # Initialize with arrays
    SurvivalData(channing.exit, entry=channing.entry, status=channing.status,
                 group=channing.sex, data=channing, warn=False)


def test_init_with_arrays():
    """Test initializing with arrays of data."""
    # Leukemia dataset
    leukemia = datasets.leukemia()
    SurvivalData(leukemia.time, status=leukemia.status, group=leukemia.group)

    # Channing House dataset
    channing = datasets.channing()
    SurvivalData(channing.exit, entry=channing.entry, status=channing.status,
                 group=channing.sex, warn=False)


def test_init_bad_data():
    """Call SurvivalData() with data=not a DataFrame."""
    with pytest.raises(TypeError):
        SurvivalData([1, 2, 3], data="not a DataFrame")


def test_init_bad_column_name():
    """Call SurvivalData() with a wrong column name."""
    channing = datasets.channing()
    with pytest.raises(ValueError):
        # The DataFrame has no column labelled "group"
        SurvivalData("exit", entry="entry", status="status", group="group",
                     data=channing)


def test_init_bad_status():
    """Call SurvivalData() with bad censoring indicators."""
    with pytest.raises(ValueError):
        SurvivalData([1, 2, 3], status=[0, 1, 2])


def test_min_time():
    """Check the min_time initializer parameter."""
    channing = datasets.channing()
    min_time = 1000
    surv = SurvivalData("exit", entry="entry", status="status", group="sex",
                        data=channing, warn=False, min_time=min_time)
    assert np.all(surv.time >= min_time)
