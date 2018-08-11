"""Unit tests for the Kaplan-Meier estimator implementation."""

import itertools
import unittest

import numpy as np
import pandas as pd

from survive.nonparametric import KaplanMeier


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
        self.assertEqual(km.data_.n_groups, 1)

        # Check estimated survival probabilities
        x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        y_true = [1., 1., 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0., 0.]
        y_true = np.reshape(y_true, (-1, 1))
        y_pred = km.predict(x)
        self.assertIsInstance(y_pred, pd.DataFrame)
        np.testing.assert_equal(y_pred.shape, (len(x), km.data_.n_groups))
        np.testing.assert_almost_equal(y_pred, y_true)

        # Try getting standard errors from predict()
        y_pred2, y_se = km.predict(x, return_se=True)
        self.assertIsInstance(y_pred2, pd.DataFrame)
        self.assertIsInstance(y_se, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred2)
        np.testing.assert_equal(y_pred2.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred2.shape, y_se.shape)

        # Try getting confidence intervals from predict()
        y_pred3, y_lower, y_upper = km.predict(x, return_ci=True)
        self.assertIsInstance(y_pred3, pd.DataFrame)
        self.assertIsInstance(y_lower, pd.DataFrame)
        self.assertIsInstance(y_upper, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred3)
        np.testing.assert_equal(y_pred3.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred3.shape, y_lower.shape)
        np.testing.assert_equal(y_pred3.shape, y_upper.shape)

        # Try getting standard errors and confidence intervals from predict()
        y_pred4, y_se, y_lower, y_upper = km.predict(x, return_se=True,
                                                     return_ci=True)
        self.assertIsInstance(y_pred4, pd.DataFrame)
        self.assertIsInstance(y_se, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred4)
        np.testing.assert_equal(y_pred4.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred4.shape, y_se.shape)
        np.testing.assert_equal(y_pred4.shape, y_lower.shape)
        np.testing.assert_equal(y_pred4.shape, y_upper.shape)

        # Check quantiles
        p = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        q_true = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.]
        q_true = np.reshape(q_true, (-1, 1))
        q_pred = km.quantile(p)
        self.assertIsInstance(q_pred, pd.DataFrame)
        np.testing.assert_almost_equal(q_pred, q_true)

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
        self.assertEqual(km.data_.n_groups, 1)

        # Check estimated survival probabilities
        x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        y_true = [1., 1., 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2]
        y_true = np.reshape(y_true, (-1, 1))
        y_pred = km.predict(x)
        self.assertIsInstance(y_pred, pd.DataFrame)
        np.testing.assert_equal(y_pred.shape, (len(x), km.data_.n_groups))
        np.testing.assert_almost_equal(y_pred, y_true)

        # Try getting standard errors from predict()
        y_pred2, y_se = km.predict(x, return_se=True)
        self.assertIsInstance(y_pred2, pd.DataFrame)
        self.assertIsInstance(y_se, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred2)
        np.testing.assert_equal(y_pred2.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred2.shape, y_se.shape)

        # Try getting confidence intervals from predict()
        y_pred3, y_lower, y_upper = km.predict(x, return_ci=True)
        self.assertIsInstance(y_pred3, pd.DataFrame)
        self.assertIsInstance(y_lower, pd.DataFrame)
        self.assertIsInstance(y_upper, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred3)
        np.testing.assert_equal(y_pred3.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred3.shape, y_lower.shape)
        np.testing.assert_equal(y_pred3.shape, y_upper.shape)

        # Try getting standard errors and confidence intervals from predict()
        y_pred4, y_se, y_lower, y_upper = km.predict(x, return_se=True,
                                                     return_ci=True)
        self.assertIsInstance(y_pred4, pd.DataFrame)
        self.assertIsInstance(y_se, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred4)
        np.testing.assert_equal(y_pred4.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred4.shape, y_se.shape)
        np.testing.assert_equal(y_pred4.shape, y_lower.shape)
        np.testing.assert_equal(y_pred4.shape, y_upper.shape)

        # Check quantiles
        p = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        q_true = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, np.nan, np.nan]
        q_true = np.reshape(q_true, (-1, 1))
        q_pred = km.quantile(p)
        self.assertIsInstance(q_pred, pd.DataFrame)
        np.testing.assert_almost_equal(q_pred, q_true)

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
        self.assertEqual(km.data_.n_groups, 2)

        # Check estimated survival probabilities
        x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        y_a = [1., 1., 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0., 0.]
        y_b = [1., 1., 1, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0.]
        y_true = np.column_stack((y_a, y_b))
        y_pred = km.predict(x)
        self.assertIsInstance(y_pred, pd.DataFrame)
        np.testing.assert_equal(y_pred.shape, (len(x), km.data_.n_groups))
        np.testing.assert_almost_equal(y_pred, y_true)

        # Try getting standard errors from predict()
        y_pred2, y_se = km.predict(x, return_se=True)
        self.assertIsInstance(y_pred2, pd.DataFrame)
        self.assertIsInstance(y_se, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred2)
        np.testing.assert_equal(y_pred2.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred2.shape, y_se.shape)

        # Try getting confidence intervals from predict()
        y_pred3, y_lower, y_upper = km.predict(x, return_ci=True)
        self.assertIsInstance(y_pred3, pd.DataFrame)
        self.assertIsInstance(y_lower, pd.DataFrame)
        self.assertIsInstance(y_upper, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred3)
        np.testing.assert_equal(y_pred3.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred3.shape, y_lower.shape)
        np.testing.assert_equal(y_pred3.shape, y_upper.shape)

        # Try getting standard errors and confidence intervals from predict()
        y_pred4, y_se, y_lower, y_upper = km.predict(x, return_se=True,
                                                     return_ci=True)
        self.assertIsInstance(y_pred4, pd.DataFrame)
        self.assertIsInstance(y_se, pd.DataFrame)
        pd.testing.assert_frame_equal(y_pred, y_pred4)
        np.testing.assert_equal(y_pred4.shape, (len(x), km.data_.n_groups))
        np.testing.assert_equal(y_pred4.shape, y_se.shape)
        np.testing.assert_equal(y_pred4.shape, y_lower.shape)
        np.testing.assert_equal(y_pred4.shape, y_upper.shape)

        # Check quantiles
        p = np.linspace(0., 1., num=11)
        q_pred = km.quantile(p)
        self.assertIsInstance(q_pred, pd.DataFrame)
        np.testing.assert_equal(q_pred.shape, (len(p), km.data_.n_groups))


if __name__ == "__main__":
    unittest.main()
