"""Unit tests for the Breslow estimator implementation."""

from itertools import product

import numpy as np

from survive import Breslow
from survive import SurvivalData
from survive import datasets

CONF_TYPES = ("linear", "log")
VAR_TYPES = ("aalen", "greenwood")
TIE_BREAKS = ("continuous", "discrete")
NA_PARAMETERS = CONF_TYPES, VAR_TYPES, TIE_BREAKS


def test_fit_predict_summary():
    """Check all the fit parameters and predictions."""
    leukemia = datasets.leukemia()
    surv = SurvivalData("time", status="status", group="group", data=leukemia)
    for conf_type, var_type, tie_break in product(*NA_PARAMETERS):
        breslow = Breslow(conf_type=conf_type, var_type=var_type,
                                   tie_break=tie_break)
        breslow.fit(surv)

        # TODO: figure out better tests here
        breslow.predict([0, 1, 2])
        breslow.predict([0, 1, 2], return_ci=True)
        breslow.summary.table("treatment")

