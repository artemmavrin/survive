"""The Kaplan-Meier estimator for non-parametric survival function estimation.

The Kaplan-Meier estimator is also called the product-limit estimator. For a
quick introduction, see Section 4.2 in Cox & Oakes (1984) or Section 1.4.1 in
Kalbfleisch & Prentice (2002).

References
----------
    * E. L. Kaplan and P. Meier. "Nonparametric estimation from incomplete
      observations". Journal of the American Statistical Association, Volume 53,
      Issue 282 (1958), pp. 457--481. doi: https://doi.org/10.2307/2281868
    * D. R. Cox and D. Oakes. Analysis of Survival Data. Chapman & Hall, London
      (1984), pp. ix+201.
    * John D. Kalbfleisch and Ross L. Prentice. The Statistical Analysis of
      Failure Time Data. Second Edition. Wiley (2002) pp. xiv+439.
"""

import numpy as np
import scipy.stats as st

from .base import NonparametricUnivariateSurvival
from ..base import SurvivalData


class KaplanMeier(NonparametricUnivariateSurvival):
    """Non-parametric survival function estimator for right-censored data.

    Properties
    ----------
    conf_type : str
        Type of confidence intervals for the survival function estimate S(t) to
        report. Possible values:
            * "plain"
                Use a normal approximation to construct confidence intervals for
                S(t) directly. The confidence interval is
                    [max(S(t) + z*SE, 0), min(S(t) - z*SE, 1)],
                where SE is the standard error estimate of S(t) computed using
                Greenwood's formula (Greenwood 1926) and z is the
                (1-conf_level)/2 quantile of the standard normal distribution.
            * "log"
                Derive confidence intervals for S(t) from normal approximation
                confidence intervals for the cumulative hazard function estimate
                -log(S(t)). The confidence interval is
                    [S(t) * exp(z*SE), min(S(t) * exp(-z*SE), 1)],
                where SE is the standard error estimate of -log(S(t)), computed
                using the delta method (similar to Greenwood's formula) and z is
                the (1-conf_level)/2 quantile of the standard normal
                distribution.
            * "log-log"
                Derive confidence intervals for S(t) from normal approximation
                confidence intervals for the log cumulative hazard function
                estimate log(-log(S(t))). The confidence interval is
                    [S(t) ** exp(z*SE), S(t) ** exp(-z*SE)],
                where SE is the standard error estimate of log(-log(S(t))),
                computed using the delta method (similar to Greenwood's
                formula) and z is the (1-conf_level)/2 quantile of the standard
                normal distribution.
    conf_level : float
        Confidence level of the confidence intervals.
    summary : KaplanMeierSummary
        A summary of this Kaplan-Meier estimator.

    References
    ----------
        * M. Greenwood. "The natural duration of cancer". Reports on Public
          Health and Medical Subjects. Volume 33 (1926), pp. 1--26.
    """

    _model_type = "Kaplan-Meier estimator"
    _conf_types = ("plain", "log", "log-log")
    _conf_type_default = "log-log"

    def fit(self, time, status=None, entry=None, group=None, data=None):
        """Fit the Kaplan-Meier estimator to survival data.

        Parameters
        ----------
        time : SurvivalData or one-dimensional array-like or str
            The observed times. If the DataFrame parameter `data` is provided,
            this can be the name of a column in `data` from which to get the
            observed times. If this is a SurvivalData instance, then all other
            parameters are ignored.
        status : one-dimensional array-like or string, optional (default: None)
            Censoring indicators. 0 means a right-censored observation, 1 means
            a true failure/event. If not provided, it is assumed that there is
            no censoring.  If the DataFrame parameter `data` is provided,
            this can be the name of a column in `data` from which to get the
            censoring indicators.
        entry : one-dimensional array-like or string, optional (default: None)
            Entry/birth times of the observations (for left-truncated data). If
            not provided, the entry time for each observation is set to 0. If
            the DataFrame parameter `data` is provided, this can be the name of
            a column in `data` from which to get the entry times.
        group : one-dimensional array-like or string, optional (default: None)
            Group/stratum labels for each observation. If not provided, the
            entire sample is taken as a single group. If the DataFrame parameter
            `data` is provided, this can be the name of a column in `data` from
            which to get the group labels.
        data : pandas.DataFrame, optional (default: None)
            Optional DataFrame from which to extract the data. If this parameter
            is specified, then the parameters `time`, `status`, `entry`, and
            `group` can be column names of this DataFrame.

        Returns
        -------
        self : KaplanMeier
            This KaplanMeier instance.
        """
        if isinstance(time, SurvivalData):
            self._data = time
        else:
            self._data = SurvivalData(time=time, status=status, entry=entry,
                                      group=group, data=data)

        # Compute the Kaplan-Meier product-limit estimator and related
        # quantities at the distinct failure times within each group
        self._survival = np.empty(self._data.n_groups, dtype=object)
        self._survival_var = np.empty(self._data.n_groups, dtype=object)
        self._survival_ci_lower = np.empty(self._data.n_groups, dtype=object)
        self._survival_ci_upper = np.empty(self._data.n_groups, dtype=object)
        for i in range(self._data.n_groups):
            e = self._data.n_events[i]
            r = self._data.n_at_risk[i]

            # Product-limit survival probability estimates
            self._survival[i] = np.cumprod(1. - e / r)

            # Sum occurring in Greenwood's formula
            with np.errstate(divide="ignore"):
                greenwood_sum = np.cumsum(e / r / (r - e))

            # Survival function variance estimates from Greenwood's formula
            with np.errstate(invalid="ignore"):
                self._survival_var[i] = (self._survival[i] ** 2) * greenwood_sum

            # Standard normal quantile for confidence intervals
            z = st.norm.ppf((1 - self.conf_level) / 2)

            # Compute confidence intervals at the observed event times
            if self._conf_type == "plain":
                # Normal approximation CI
                c = z * np.sqrt(self._survival_var[i])
                lower = self._survival[i] + c
                upper = self._survival[i] - c
            elif self._conf_type == "log":
                # CI based on a CI for the cumulative hazard -log(S(t))
                with np.errstate(invalid="ignore"):
                    c = z * np.sqrt(greenwood_sum)
                    lower = self._survival[i] * np.exp(c)
                    upper = self._survival[i] * np.exp(-c)
            elif self._conf_type == "log-log":
                # CI based on a CI for log(-log(S(t)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_s = np.log(self._survival[i])
                    c = z * np.sqrt(greenwood_sum) / log_s
                    lower = self._survival[i] ** (np.exp(c))
                    upper = self._survival[i] ** (np.exp(-c))
            else:
                # This should not be reachable
                raise ValueError(
                    f"Invalid confidence interval type: {self._conf_type}.")

            # Force confidence interval bounds to be between 0 and 1
            with np.errstate(invalid="ignore"):
                self._survival_ci_lower[i] = np.maximum(lower, 0.)
                self._survival_ci_upper[i] = np.minimum(upper, 1.)

        self.fitted = True
        return self
