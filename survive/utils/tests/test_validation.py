"""Unit tests for the validation functions."""

import numpy as np
import pandas as pd
import pytest

from survive.utils import check_bool
from survive.utils import check_data_1d
from survive.utils import check_data_2d
from survive.utils import check_float
from survive.utils import check_int
from survive.utils import check_random_state


def test_check_bool():
    """Sanity checks for check_bool()."""
    assert check_bool(True)
    assert not check_bool(False)
    assert check_bool(None, allow_none=True) is None
    for bad in (None, 0, 1, "a", [1, 2, 3], []):
        with pytest.raises(TypeError):
            check_bool(bad)


def test_check_int():
    """Sanity checks for check_int()."""
    assert check_int(1) == 1
    assert check_int(np.int_(2)) == np.int_(2)
    assert check_int(None, allow_none=True) is None
    for bad in (None, 0., 1., "a", [1, 2, 3], []):
        with pytest.raises(TypeError):
            check_int(bad)

    # Check that min/max constraints are enforced
    assert check_int(0, minimum=0) == 0
    assert check_int(1, minimum=0) == 1
    with pytest.raises(ValueError):
        check_int(-1, minimum=0)
    assert check_int(0, maximum=1) == 0
    assert check_int(1, maximum=1) == 1
    with pytest.raises(ValueError):
        check_int(2, maximum=1)


def test_check_float():
    """Sanity checks for check_float()."""
    assert check_float(1.) == 1.
    assert check_float(np.float_(2.)) == np.float_(2.)
    assert check_float(0) == 0
    assert check_float(None, allow_none=True) is None
    for bad in (None, "a", [1, 2, 3], []):
        with pytest.raises(TypeError):
            check_float(bad)

    # Check that min/max/positivity constraints are enforced
    assert check_float(0, minimum=0) == 0
    assert check_float(1, minimum=0) == 1
    with pytest.raises(ValueError):
        check_float(-1, minimum=0)
    assert check_float(0, maximum=1) == 0
    assert check_float(1, maximum=1) == 1
    with pytest.raises(ValueError):
        check_float(2, maximum=1)
    assert check_float(1., positive=True) == 1.
    assert check_float(1e-8, positive=True) == 1e-8
    with pytest.raises(ValueError):
        check_float(0, positive=True)


def test_check_data_1d():
    """Sanity checks for check_data_1d()."""
    # One-dimensional arrays
    for x in ([0], [1, 2, 3], np.arange(10)):
        y = check_data_1d(x)
        np.testing.assert_equal(y, x)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

    # Check pandas.Series with keep_pandas=True
    series = pd.Series([1., 2., 3.])
    y = check_data_1d(series)
    assert isinstance(y, pd.Series)
    pd.testing.assert_series_equal(y, series)

    # Check pandas.Series with keep_pandas=False
    y = check_data_1d(series, keep_pandas=False)
    assert isinstance(y, np.ndarray)
    np.testing.assert_equal(y, series)

    # Scalars should be coerced into 1D arrays
    x = 0
    y = check_data_1d(x)
    np.testing.assert_equal(y, [x])
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1

    # Empty, 2D, and higher-dimensional arrays should raise errors
    for x in ([], [[1], [2]], [[0, 1], [2, 3]], np.ones((3, 4, 5))):
        with pytest.raises(ValueError):
            check_data_1d(x)

    # A TypeError should be raised if a non-numeric array is passed
    non_numeric = ["this", "isn't", "a", "numeric", "array"]
    with pytest.raises(TypeError):
        check_data_1d(non_numeric)

    # A non-numeric array should go through if numeric=False
    check_data_1d(non_numeric, numeric=False)

    # Check that size constraints are enforced
    x = [1, 2, 3]
    np.testing.assert_equal(check_data_1d(x, n_min=1), x)
    np.testing.assert_equal(check_data_1d(x, n_max=4), x)
    np.testing.assert_equal(check_data_1d(x, n_exact=3), x)
    for constraint in (dict(n_min=4), dict(n_max=2), dict(n_exact=1)):
        with pytest.raises(ValueError):
            check_data_1d(x, **constraint)


def test_check_data_2d():
    """Sanity checks for check_data_2d()."""
    # Check pandas.DataFrames with keep_pandas=True
    df = pd.DataFrame([[0, 1, 2], [3, 4, 5]])
    y = check_data_2d(df)
    assert isinstance(y, pd.DataFrame)
    pd.testing.assert_frame_equal(y, df)

    # Check pandas.DataFrames with keep_pandas=False
    y = check_data_2d(df, keep_pandas=False)
    assert isinstance(y, np.ndarray)
    np.testing.assert_equal(y, df)

    # Two-dimensional arrays
    for x in ([[0]], [[0], [1]], [[0, 1]], [[0, 1], [2, 3]]):
        y = check_data_2d(x)
        np.testing.assert_equal(y, x)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 2
        assert y.shape == np.shape(x)

    # One-dimensional arrays
    for x in ([0], [1, 2, 3], np.arange(10)):
        y = check_data_2d(x)
        np.testing.assert_equal(y, np.reshape(x, (-1, 1)))
        assert isinstance(y, np.ndarray)
        assert y.ndim == 2
        assert y.shape == (len(x), 1)

    # Scalars should be coerced into 1D arrays
    x = 0
    y = check_data_2d(x)
    np.testing.assert_equal(y, [[x]])
    assert isinstance(y, np.ndarray)
    assert y.ndim == 2

    # Empty, 3D, and higher-dimensional arrays should raise errors
    for x in ([], [[[1]], [[2]]], [[[[4]]]], np.ones((3, 4, 5))):
        with pytest.raises(ValueError):
            check_data_2d(x)

    # A TypeError should be raised if a non-numeric array is passed
    non_numeric = ["this", "isn't", "a", "numeric", "array"]
    with pytest.raises(TypeError):
        check_data_2d(non_numeric)

    # A non-numeric array should go through if numeric=False
    check_data_2d(non_numeric, numeric=False)

    # A TypeError should be raised if a non-numeric DataFrame is passed
    non_numeric_df = pd.DataFrame([dict(a=1, b="a"), dict(a=2, b="c")])
    with pytest.raises(TypeError):
        check_data_2d(non_numeric_df)

    # A non-numeric DataFrame should go through if numeric=False
    check_data_2d(non_numeric_df, numeric=False)

    # Check that size constraints are enforced
    x = [[1], [2], [3]]
    np.testing.assert_equal(check_data_2d(x, n_min=1), x)
    np.testing.assert_equal(check_data_2d(x, n_max=4), x)
    np.testing.assert_equal(check_data_2d(x, n_exact=3), x)
    np.testing.assert_equal(check_data_2d(x, p_min=1), x)
    np.testing.assert_equal(check_data_2d(x, p_max=2), x)
    np.testing.assert_equal(check_data_2d(x, p_exact=1), x)
    constraints = (dict(n_min=4), dict(n_max=2), dict(n_exact=1),
                   dict(p_min=2), dict(p_exact=3), dict(p_max=0))
    for constraint in constraints:
        with pytest.raises(ValueError):
            check_data_2d(x, **constraint)


def test_check_random_state():
    """Sanity checks for check_random_state()."""
    for seed in (None, 0, np.random.RandomState(1)):
        random_state = check_random_state(seed)
        assert isinstance(random_state, np.random.RandomState)
        if isinstance(seed, np.random.RandomState):
            assert random_state is seed
