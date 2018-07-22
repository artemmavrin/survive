"""Unit tests for the Kaplan-Meier estimator implementation."""

import itertools
import unittest

import numpy as np
import pandas as pd

from survive.univariate import KaplanMeier


class TestKaplanMeier(unittest.TestCase):
    def test_no_censoring_one_group(self):
        """Simple example with one group and no censoring.

        The data are the numbers 1, 2, 3, 4, 5. In the absence of censoring, the
        Kaplan-Meier estimate of the survival function should coincide with the
        empirical survival function, shown below.

                |
            1.0 |-----+
                |     |
            0.8 |     +-----+
                |           |
            0.6 |           +-----+
                |                 |
            0.4 |                 +-----+
                |                       |
            0.2 |                       +-----+
                |                             |
            0.0 +-----+-----+-----+-----+-----+-----
                0     1     2     3     4     5

        This test just verifies this and does some other sanity checks.
        """
        time = [1, 2, 3, 4, 5]
        km = KaplanMeier()
        km.fit(time)

        # There should only be one group
        self.assertEqual(km.data.n_groups, 1)

        # Check estimated survival probabilities
        data = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        survival = [1., 1., 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0., 0.]
        np.testing.assert_almost_equal(km.predict(data), survival)

        # Check quantiles
        prob = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        quantiles = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.]
        np.testing.assert_almost_equal(km.quantile(prob), quantiles)

        # Calling predict(), var(), se() with an array argument should return
        # an array. Calling ci() should return a pair of arrays.
        lower, upper = km.ci(data)
        for val in (km.predict(data), km.var(data), km.se(data), lower, upper):
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (len(data),))

        # Calling predict(), var(), se() with a scalar argument should return
        # a float. Calling ci() should return a pair of floats.
        for t in data:
            lower, upper = km.ci(t)
            for val in (km.predict(t), km.var(t), km.se(t), lower, upper):
                self.assertIsInstance(val, float)

    def test_censoring_one_group(self):
        """Simple example with one group and censoring.

        The data are the numbers 1, 2, 3, 4, 5+ (the + indicates that
        observation is censored). In this easy case, the Kaplan-Meier estimator
        should look like

                |
            1.0 |-----+
                |     |
            0.8 |     +-----+
                |           |
            0.6 |           +-----+
                |                 |
            0.4 |                 +-----+
                |                       |
            0.2 |                       +-----------...
                |
            0.0 +-----+-----+-----+-----+-----+-----
                0     1     2     3     4     5
        """
        time = [1, 2, 3, 4, 5]
        status = [1, 1, 1, 1, 0]
        km = KaplanMeier()
        km.fit(time, status=status)

        # There should only be one group
        self.assertEqual(km.data.n_groups, 1)

        # Check estimated survival probabilities
        data = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        survival \
            = [1., 1., 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2]
        np.testing.assert_almost_equal(km.predict(data), survival)

        # Check quantiles
        prob = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        quantiles = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, np.nan, np.nan]
        np.testing.assert_almost_equal(km.quantile(prob), quantiles)

        # Calling predict(), var(), se() with an array argument should return
        # an array. Calling ci() should return a pair of arrays.
        lower, upper = km.ci(data)
        for val in (km.predict(data), km.var(data), km.se(data), lower, upper):
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (len(data),))

        # Calling predict(), var(), se() with a scalar argument should return
        # a float. Calling ci() should return a pair of floats.
        for t in data:
            lower, upper = km.ci(t)
            for val in (km.predict(t), km.var(t), km.se(t), lower, upper):
                self.assertIsInstance(val, float)

    def test_no_censoring_two_groups(self):
        """Simple example with two group and no censoring.

        Group a: 1, 2, 3, 4, 5
                |
            1.0 |-----+
                |     |
            0.8 |     +-----+
                |           |
            0.6 |           +-----+
                |                 |
            0.4 |                 +-----+
                |                       |
            0.2 |                       +-----+
                |                             |
            0.0 +-----+-----+-----+-----+-----+-----
                0     1     2     3     4     5

        Group b: 1.5, 2.5, 3.5, 4.5, 5.5
                |
            1.0 |--------+
                |        |
            0.8 |        +-----+
                |              |
            0.6 |              +-----+
                |                    |
            0.4 |                    +-----+
                |                          |
            0.2 |                          +-----+
                |                                |
            0.0 +--+-----+-----+-----+-----+-----+--
                0 0.5   1.5   2.5   3.5   4.5   5.5
        """
        time = [1, 2, 3, 4, 5, 1.5, 2.5, 3.5, 4.5, 5.5]
        group = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]
        km = KaplanMeier()
        km.fit(time, group=group)

        # There should be two groups
        self.assertEqual(km.data.n_groups, 2)

        # Check estimated survival probabilities for each group
        data = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        survival0 = [1., 1., 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0., 0.]
        survival1 = [1., 1., 1, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0.]
        np.testing.assert_almost_equal(km.predict(data, "a"), survival0)
        np.testing.assert_almost_equal(km.predict(data, "b"), survival1)

        # Check estimated survival probabilities for both groups at once
        prob = km.predict(data)
        self.assertIsInstance(prob, pd.DataFrame)
        np.testing.assert_equal(prob.columns.values, ["a", "b"])
        np.testing.assert_almost_equal(prob["a"], survival0)
        np.testing.assert_almost_equal(prob["b"], survival1)

        # Calling predict(), var(), se() with an array argument should return
        # a DataFrame. Calling ci() should return a pair of DataFrames.
        lower, upper = km.ci(data)
        for val in (km.predict(data), km.var(data), km.se(data), lower, upper):
            self.assertIsInstance(val, pd.DataFrame)
            np.testing.assert_equal(val.columns.values, ["a", "b"])
            self.assertEqual(val.shape, (len(data), 2))

        # Calling predict(), var(), se() with a scalar argument should return
        # a DataFrame. Calling ci() should return a pair of DataFrames.
        for t in data:
            lower, upper = km.ci(t)
            for val in (km.predict(t), km.var(t), km.se(t), lower, upper):
                self.assertIsInstance(val, pd.DataFrame)
                np.testing.assert_equal(val.columns.values, ["a", "b"])
                self.assertEqual(val.shape, (1, 2))

        # Calling predict(), var(), se() with an array argument and specifying a
        # group should return an array Calling ci() should return a pair of
        # arrays.
        for group in ("a", "b"):
            lower, upper = km.ci(data, group=group)
            for val in (km.predict(data, group=group),
                        km.var(data, group=group), km.se(data, group=group),
                        lower, upper):
                self.assertIsInstance(val, np.ndarray)
                self.assertEqual(val.shape, (len(data),))

        # Calling predict(), var(), se() with a scalar argument and a specified
        # group should return a float. Calling ci() should return a pair of
        # floats.
        for group, t in itertools.product(("a", "b"), data):
            lower, upper = km.ci(t, group=group)
            for val in (km.predict(t, group=group), km.var(t, group=group),
                        km.se(t, group=group), lower, upper):
                self.assertIsInstance(val, float)


if __name__ == "__main__":
    unittest.main()
