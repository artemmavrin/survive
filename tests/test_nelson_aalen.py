"""Unit tests for the Nelson-Aalen estimator implementation."""

from itertools import product

import numpy as np

from survive import NelsonAalen
from survive import SurvivalData
from survive import datasets

CONF_TYPES = ("linear", "log")
VAR_TYPES = ("aalen", "greenwood")
TIE_BREAKS = ("continuous", "discrete")
NA_PARAMETERS = CONF_TYPES, VAR_TYPES, TIE_BREAKS


def test_leukemia():
    """Check the Nelson-Aalen estimator on the leukemia dataset."""
    leukemia = datasets.leukemia()
    nelson_aalen = NelsonAalen(var_type="aalen", tie_break="discrete")
    nelson_aalen.fit("time", status="status", group="group", data=leukemia)

    # Table 4.2 on p. 94 in  in Klein & Moeschberger (2003) displays the
    # Nelson-Aalen cumulative hazard estimates and standard errors for the
    # treatment group to 4 decimal places
    times = np.array([0, 6, 7, 10, 13, 16, 22, 23])
    cum_haz_treatment = \
        [0., 0.1428, 0.2017, 0.2683, 0.3517, 0.4426, 0.5854, 0.7521]
    std_err_treatment = [0., 0.0825, 0.1015, 0.1212, 0.1473, 0.1729, 0.2243,
                         0.2795]

    for eps in (0., 0.5):
        # Perturb the times forward by a small amount `eps` to ensure that
        # the estimates are right continuous piecewise constant
        cum_haz_pred, std_err_pred = \
            nelson_aalen.predict(times + eps, return_se=True)

        np.testing.assert_almost_equal(cum_haz_pred.treatment,
                                       cum_haz_treatment, decimal=3)

        np.testing.assert_almost_equal(std_err_pred.treatment,
                                       std_err_treatment, decimal=3)


def test_fit_predict():
    """Check all the fit parameters and predictions."""
    leukemia = datasets.leukemia()
    surv = SurvivalData("time", status="status", group="group", data=leukemia)
    for conf_type, var_type, tie_break in product(*NA_PARAMETERS):
        nelson_aalen = NelsonAalen(conf_type=conf_type, var_type=var_type,
                                   tie_break=tie_break)
        nelson_aalen.fit(surv)

        # TODO: figure out better tests here
        nelson_aalen.predict([0, 1, 2])
        nelson_aalen.predict([0, 1, 2], return_ci=True)
        nelson_aalen.summary.table("treatment")

