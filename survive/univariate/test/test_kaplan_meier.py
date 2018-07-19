"""Unit tests for the Kaplan-Meier estimator implementation."""

import unittest

import numpy as np

from survive.univariate import KaplanMeier


class TestKaplanMeier(unittest.TestCase):
    def test_no_censoring_one_group(self):
        """Simple example with one group and no censoring.

        The data are the numbers 1, 2, 3, 4, 5. In the absence of censoring, the
        Kaplan-Meier estimate of the survival function should coincide with the
        empirical survival function, shown below.

                |
            1.0 |----+
                |    |
            0.8 |    +----+
                |         |
            0.6 |         +----+
                |              |
            0.4 |              +----+
                |                   |
            0.2 |                   +----+
                |                        |
            0.0 +----+----+----+----+----+----
                0    1    2    3    4    5

        This test just verifies this and does some other sanity checks.
        """
        time = [1, 2, 3, 4, 5]
        km = KaplanMeier()
        km.fit(time)

        # Check estimated survival probabilities
        data = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        survival = [1., 1., 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0., 0.]
        np.testing.assert_almost_equal(km.predict(data), survival)

        # Calling predict(), var(), se() with an array argument should return
        # an array. Calling ci() should return a pair of arrays.
        lower, upper = km.ci(data)
        for val in (km.predict(data), km.var(data), km.se(data), lower, upper):
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (len(data),))

        # There should only be one group
        self.assertEqual(km.data.n_groups, 1)

        # Calling predict(), var(), se() with a scalar argument should return
        # a float. Calling ci() should return a pair of floats.
        for t in data:
            lower, upper = km.ci(t)
            for val in (km.predict(t), km.var(t), km.se(t), lower, upper):
                self.assertIsInstance(val, float)


if __name__ == "__main__":
    unittest.main()
