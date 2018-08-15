"""Unit tests for the Kaplan-Meier estimator implementation."""

import unittest
from itertools import product

import numpy as np
import pandas as pd

from survive import KaplanMeier
from survive import datasets

CONF_TYPES = ("arcsin", "linear", "log", "log-log", "logit")
VAR_TYPES = ("aalen-johansen", "bootstrap", "greenwood")
TIE_BREAKS = ("continuous", "discrete")
KM_PARAMETERS = CONF_TYPES, VAR_TYPES, TIE_BREAKS


class TestKaplanMeier(unittest.TestCase):
    """Unit tests for the Kaplan-Meier estimator implementation."""

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

        for conf_type, var_type, tie_break in product(*KM_PARAMETERS):
            # Initialize the Kaplan-Meier estimator
            kaplan_meier = KaplanMeier(conf_type=conf_type, var_type=var_type,
                                       tie_break=tie_break, random_state=0)

            # Fit the Kaplan-Meier estimator
            kaplan_meier.fit(time)

            # There should only be one group
            self.assertEqual(kaplan_meier.data_.n_groups, 1)

            # Check estimated survival probabilities
            x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
            y_true = [1., 1., 0.8, 0.8, 0.6, 0.6,
                      0.4, 0.4, 0.2, 0.2, 0., 0., 0.]
            y_true = np.reshape(y_true, (-1, 1))
            y_pred = kaplan_meier.predict(x)
            self.assertIsInstance(y_pred, pd.DataFrame)
            np.testing.assert_equal(y_pred.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_almost_equal(y_pred, y_true)

            # Get standard errors from predict()
            y_pred, y_se = kaplan_meier.predict(x, return_se=True)
            self.assertIsInstance(y_pred, pd.DataFrame)
            self.assertIsInstance(y_se, pd.DataFrame)
            np.testing.assert_almost_equal(y_pred, y_true)
            np.testing.assert_equal(y_pred.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred.shape, y_se.shape)

            # Get confidence intervals from predict()
            y_pred, y_lower, y_upper = kaplan_meier.predict(x, return_ci=True)
            self.assertIsInstance(y_pred, pd.DataFrame)
            self.assertIsInstance(y_lower, pd.DataFrame)
            self.assertIsInstance(y_upper, pd.DataFrame)
            np.testing.assert_almost_equal(y_pred, y_true)
            np.testing.assert_equal(y_pred.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred.shape, y_lower.shape)
            np.testing.assert_equal(y_pred.shape, y_upper.shape)

            # Get standard errors and confidence intervals from predict()
            y_pred, y_se, y_lower, y_upper = \
                kaplan_meier.predict(x, return_se=True, return_ci=True)
            self.assertIsInstance(y_pred, pd.DataFrame)
            self.assertIsInstance(y_se, pd.DataFrame)
            np.testing.assert_almost_equal(y_pred, y_true)
            np.testing.assert_equal(y_pred.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred.shape, y_se.shape)
            np.testing.assert_equal(y_pred.shape, y_lower.shape)
            np.testing.assert_equal(y_pred.shape, y_upper.shape)

            # Check quantiles
            p = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
            q_true = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.]
            q_true = np.reshape(q_true, (-1, 1))
            q_pred = kaplan_meier.quantile(p)
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

        for conf_type, var_type, tie_break in product(*KM_PARAMETERS):
            # Initialize the Kaplan-Meier estimator
            kaplan_meier = KaplanMeier(conf_type=conf_type, var_type=var_type,
                                       tie_break=tie_break, random_state=0)

            # Fit the Kaplan-Meier estimator
            kaplan_meier.fit(time, status=status)

            # There should only be one group
            self.assertEqual(kaplan_meier.data_.n_groups, 1)

            # Check estimated survival probabilities
            x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
            y_true = [1., 1., 0.8, 0.8, 0.6, 0.6,
                      0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2]
            y_true = np.reshape(y_true, (-1, 1))
            y_pred = kaplan_meier.predict(x)
            self.assertIsInstance(y_pred, pd.DataFrame)
            np.testing.assert_equal(y_pred.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_almost_equal(y_pred, y_true)

            # Get standard errors from predict()
            y_pred2, y_se = kaplan_meier.predict(x, return_se=True)
            self.assertIsInstance(y_pred2, pd.DataFrame)
            self.assertIsInstance(y_se, pd.DataFrame)
            pd.testing.assert_frame_equal(y_pred, y_pred2)
            np.testing.assert_equal(y_pred2.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred2.shape, y_se.shape)

            # Get confidence intervals from predict()
            y_pred3, y_lower, y_upper = kaplan_meier.predict(x, return_ci=True)
            self.assertIsInstance(y_pred3, pd.DataFrame)
            self.assertIsInstance(y_lower, pd.DataFrame)
            self.assertIsInstance(y_upper, pd.DataFrame)
            pd.testing.assert_frame_equal(y_pred, y_pred3)
            np.testing.assert_equal(y_pred3.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred3.shape, y_lower.shape)
            np.testing.assert_equal(y_pred3.shape, y_upper.shape)

            # Get standard errors and confidence intervals from predict()
            y_pred4, y_se, y_lower, y_upper = \
                kaplan_meier.predict(x, return_se=True, return_ci=True)
            self.assertIsInstance(y_pred4, pd.DataFrame)
            self.assertIsInstance(y_se, pd.DataFrame)
            pd.testing.assert_frame_equal(y_pred, y_pred4)
            np.testing.assert_equal(y_pred4.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred4.shape, y_se.shape)
            np.testing.assert_equal(y_pred4.shape, y_lower.shape)
            np.testing.assert_equal(y_pred4.shape, y_upper.shape)

            # Check quantiles
            p = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
            q_true = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, np.nan, np.nan]
            q_true = np.reshape(q_true, (-1, 1))
            q_pred = kaplan_meier.quantile(p)
            self.assertIsInstance(q_pred, pd.DataFrame)
            np.testing.assert_almost_equal(q_pred, q_true)

    def test_no_censoring_two_groups(self):
        """Simple example with two groups and no censoring.

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

        for conf_type, var_type, tie_break in product(*KM_PARAMETERS):
            # Initialize the Kaplan-Meier estimator
            kaplan_meier = KaplanMeier(conf_type=conf_type, var_type=var_type,
                                       tie_break=tie_break, random_state=0)

            # Fit the Kaplan-Meier estimator
            kaplan_meier.fit(time, group=group)

            # There should be two groups
            self.assertEqual(kaplan_meier.data_.n_groups, 2)

            # Check estimated survival probabilities
            x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
            y_a = [1., 1., 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0., 0.]
            y_b = [1., 1., 1, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2, 0., 0.]
            y_true = np.column_stack((y_a, y_b))
            y_pred = kaplan_meier.predict(x)
            self.assertIsInstance(y_pred, pd.DataFrame)
            np.testing.assert_equal(y_pred.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_almost_equal(y_pred, y_true)

            # Get standard errors from predict()
            y_pred2, y_se = kaplan_meier.predict(x, return_se=True)
            self.assertIsInstance(y_pred2, pd.DataFrame)
            self.assertIsInstance(y_se, pd.DataFrame)
            pd.testing.assert_frame_equal(y_pred, y_pred2)
            np.testing.assert_equal(y_pred2.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred2.shape, y_se.shape)

            # Get confidence intervals from predict()
            y_pred3, y_lower, y_upper = kaplan_meier.predict(x, return_ci=True)
            self.assertIsInstance(y_pred3, pd.DataFrame)
            self.assertIsInstance(y_lower, pd.DataFrame)
            self.assertIsInstance(y_upper, pd.DataFrame)
            pd.testing.assert_frame_equal(y_pred, y_pred3)
            np.testing.assert_equal(y_pred3.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred3.shape, y_lower.shape)
            np.testing.assert_equal(y_pred3.shape, y_upper.shape)

            # Get standard errors and confidence intervals from predict()
            y_pred4, y_se, y_lower, y_upper = \
                kaplan_meier.predict(x, return_se=True, return_ci=True)
            self.assertIsInstance(y_pred4, pd.DataFrame)
            self.assertIsInstance(y_se, pd.DataFrame)
            pd.testing.assert_frame_equal(y_pred, y_pred4)
            np.testing.assert_equal(y_pred4.shape,
                                    (len(x), kaplan_meier.data_.n_groups))
            np.testing.assert_equal(y_pred4.shape, y_se.shape)
            np.testing.assert_equal(y_pred4.shape, y_lower.shape)
            np.testing.assert_equal(y_pred4.shape, y_upper.shape)

            # Check quantiles
            p = np.linspace(0., 1., num=11)
            q_pred = kaplan_meier.quantile(p)
            self.assertIsInstance(q_pred, pd.DataFrame)
            np.testing.assert_equal(q_pred.shape,
                                    (len(p), kaplan_meier.data_.n_groups))

    def test_leukemia(self):
        """Check computed values on the leukemia dataset."""
        leukemia = datasets.leukemia()
        kaplan_meier = KaplanMeier(var_type="greenwood")
        kaplan_meier.fit("time", status="status", group="group", data=leukemia)

        # Table Table 4.1 on p. 49 in Cox & Oakes (1984) displays the
        # Kaplan-Meier estimates for the treatment group to 4 decimal places,
        # and Table 4.1B on p. 93 in Klein & Moeschberger (2003) lists their
        # Greenwood's formula-based standard errors to 3 decimal places
        times = np.array([0, 6, 7, 10, 13, 16, 22, 23])
        survival_treatment = \
            [1., 0.8571, 0.8067, 0.7529, 0.6902, 0.6275, 0.5378, 0.4482]
        std_err_treatment = \
            [0., 0.076, 0.087, 0.096, 0.107, 0.114, 0.128, 0.135]

        for eps in (0., 0.5):
            # Perturb the times forward by a small amount `eps` to ensure that
            # the estimates are right continuous piecewise constant
            survival_pred, std_err_pred = \
                kaplan_meier.predict(times + eps, return_se=True)

            np.testing.assert_almost_equal(survival_pred.treatment,
                                           survival_treatment, decimal=3)

            np.testing.assert_almost_equal(std_err_pred.treatment,
                                           std_err_treatment, decimal=3)

        # The Example in the left margin of p. 53 in Kleinbaum & Klein (2005)
        # lists the Kaplan-Meier estimates for the control group. Since there is
        # no censoring in the control group, this is the same as the empirical
        # survival function
        times = np.asarray([0, 1, 2, 3, 4, 5, 8, 11, 12, 15, 17, 22, 23])
        survival_control = [1., 19 / 21, 17 / 21, 16 / 21, 14 / 21, 12 / 21,
                            8 / 21, 6 / 21, 4 / 21, 3 / 21, 2 / 21, 1 / 21, 0.]

        for eps in (0., 0.5):
            # Perturb the times forward by a small amount `eps` to ensure that
            # the estimates are right continuous piecewise constant
            survival_pred, std_err_pred = \
                kaplan_meier.predict(times + eps, return_se=True)

            np.testing.assert_almost_equal(survival_pred.control,
                                           survival_control, decimal=3)

        # Page 27 in http://www.math.ucsd.edu/~rxu/math284/slect2.pdf lists
        # Kaplan-Meier summary tables from R's survfit function (in the survival
        # package) with three different types of confidence intervals for the
        # treatment group
        times = np.array([6, 7, 10, 13, 16, 22, 23])

        # Confidence intervals of type "log"
        ci_lower_log = [0.7198171, 0.6531242, 0.5859190, 0.5096131, 0.4393939,
                        0.3370366, 0.2487882]
        ci_upper_log = [1.0000000, 0.9964437, 0.9675748, 0.9347692, 0.8959949,
                        0.8582008, 0.8073720]

        kaplan_meier = KaplanMeier(conf_type="log", var_type="greenwood")
        kaplan_meier.fit("time", status="status", group="group", data=leukemia)

        for eps in (0., 0.5):
            # Perturb the times forward by a small amount `eps` to ensure that
            # the estimates are right continuous piecewise constant
            _, ci_lower_pred, ci_upper_pred = \
                kaplan_meier.predict(times + eps, return_ci=True)

            np.testing.assert_almost_equal(ci_lower_pred.treatment,
                                           ci_lower_log, decimal=7)
            np.testing.assert_almost_equal(ci_upper_pred.treatment,
                                           ci_upper_log, decimal=7)

        # Confidence intervals of type "log-log"
        ci_lower_log_log = [0.6197180, 0.5631466, 0.5031995, 0.4316102,
                            0.3675109, 0.2677789, 0.1880520]
        ci_upper_log_log = [0.9515517, 0.9228090, 0.8893618, 0.8490660,
                            0.8049122, 0.7467907, 0.6801426]

        kaplan_meier = KaplanMeier(conf_type="log-log", var_type="greenwood")
        kaplan_meier.fit("time", status="status", group="group", data=leukemia)

        for eps in (0., 0.5):
            # Perturb the times forward by a small amount `eps` to ensure that
            # the estimates are right continuous piecewise constant
            _, ci_lower_pred, ci_upper_pred = \
                kaplan_meier.predict(times + eps, return_ci=True)

            np.testing.assert_almost_equal(ci_lower_pred.treatment,
                                           ci_lower_log_log, decimal=7)
            np.testing.assert_almost_equal(ci_upper_pred.treatment,
                                           ci_upper_log_log, decimal=7)

        # Confidence intervals of type "linear" (in R: "plain")
        ci_lower_linear = [0.7074793, 0.6363327, 0.5640993, 0.4808431,
                           0.4039095, 0.2864816, 0.1843849]
        ci_upper_linear = [1.0000000, 0.9771127, 0.9417830, 0.8995491,
                           0.8509924, 0.7891487, 0.7119737]

        kaplan_meier = KaplanMeier(conf_type="linear", var_type="greenwood")
        kaplan_meier.fit("time", status="status", group="group", data=leukemia)

        for eps in (0., 0.5):
            # Perturb the times forward by a small amount `eps` to ensure that
            # the estimates are right continuous piecewise constant
            _, ci_lower_pred, ci_upper_pred = \
                kaplan_meier.predict(times + eps, return_ci=True)

            np.testing.assert_almost_equal(ci_lower_pred.treatment,
                                           ci_lower_linear, decimal=7)
            np.testing.assert_almost_equal(ci_upper_pred.treatment,
                                           ci_upper_linear, decimal=7)


if __name__ == "__main__":
    unittest.main()
