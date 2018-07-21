"""The Kaplan-Meier non-parametric survival function estimator.

The Kaplan-Meier estimator (Kaplan & Meier 1958) is also called the
product-limit estimator. Much of this implementation is inspired by the R
package ``survival`` (Therneau (2015)).

For a quick introduction to the Kaplan-Meier estimator, see e.g. Section 4.2 in
Cox & Oakes (1984) or Section 1.4.1 in Kalbfleisch & Prentice (2002). For a more
thorough treatment, see Chapter 4 in Klein & Moeschberger (2003).

References
----------
    * E. L. Kaplan and P. Meier. "Nonparametric estimation from incomplete
      observations". Journal of the American Statistical Association, Volume 53,
      Issue 282 (1958), pp. 457--481. doi: https://doi.org/10.2307/2281868
    * Terry M. Therneau. A Package for Survival Analysis in S. version 2.38
      (2015). CRAN: https://CRAN.R-project.org/package=survival
    * D. R. Cox and D. Oakes. Analysis of Survival Data. Chapman & Hall, London
      (1984), pp. ix+201.
    * John D. Kalbfleisch and Ross L. Prentice. The Statistical Analysis of
      Failure Time Data. Second Edition. Wiley (2002) pp. xiv+439.
    * John P. Klein and Melvin L. Moeschberger. Survival Analysis. Techniques
      for Censored and Truncated Data. Second Edition. Springer-Verlag, New York
      (2003) pp. xvi+538. doi: https://doi.org/10.1007/b97377
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
        Type of confidence interval for the survival function estimate to
        report. Possible values:
            * "linear"
            * "log"
            * "log-log"
            * "logit"
            * "arcsin"
        Confidence intervals for a survival probability p=S(t) are computed
        using normal approximation confidence intervals for a strictly
        increasing differentiable transformation y=f(p) using the delta method:
        if se(p) is the standard error of p, then the standard error of f(p) is
        se(p)*f'(p). Consequently, a confidence interval for f(p) is
            [f(p) + z * se(p) * f'(p), f(p) - z * se(p) * f'(p)],
        where z is the (1-conf_level)/2-quantile of the standard normal
        distribution. If g(y) denotes the inverse of f, then a confidence
        interval for p is
                [g(f(p) + z * se(p) * f'(p)), g(f(p) - z * se(p) * f'(p))].
        These confidence intervals were proposed by Borgan & Liestøl (1990). We
        give a table of the supported transformations below.

            name        f(p)            f'(p)               g(y)
            ------------------------------------------------------------------
            "linear"    p               1                   y
            "log"       log(p)          1/p                 exp(y)
            "log-log"   -log(-log(p))   -1/(p*log(p))       exp(-exp(-y))
            "logit"     log(p/(1-p))    1/(p*(1-p))         exp(y)/(1+exp(y))
            "arcsin"    arcsin(sqrt(p)) 1/(2*sqrt(p*(1-p))) sin(y)**2

        Our implementation also shrinks the intervals to be between 0 and 1 if
        necessary.
    conf_level : float
        Confidence level of the confidence intervals.
    var_type : str
        Type of variance estimate for the survival function to compute.
        Possible values:
            * "greenwood"
                Use Greenwood's formula (Greenwood (1926)).
            * "aalen-johansen"
                Use the variance estimate suggested by Aalen & Johansen (1978).

    References
    ----------
        * Ørnulf Borgan and Knut Liestøl. "A note on confidence intervals and
          bands for the survival function based on transformations."
          Scandinavian Journal of Statistics. Volume 17, Number 1 (1990),
          pp. 35--41. JSTOR: http://www.jstor.org/stable/4616153
        * M. Greenwood. "The natural duration of cancer". Reports on Public
          Health and Medical Subjects. Volume 33 (1926), pp. 1--26.
        * Odd O. Aalen and Søren Johansen. "An empirical transition matrix for
          non-homogeneous Markov chains based on censored observations."
          Scandinavian Journal of Statistics. Volume 5, Number 3 (1978),
          pp. 141--150. JSTOR: http://www.jstor.org/stable/4615704
    """

    _model_type = "Kaplan-Meier estimator"
    _conf_types = ("linear", "log", "log-log", "logit", "arcsin")

    # Types of variance estimators
    _var_types = ("greenwood", "aalen-johansen")
    _var_type: str

    @property
    def var_type(self):
        """Type of variance estimate for the survival function to compute."""
        return self._var_type

    @var_type.setter
    def var_type(self, var_type):
        """Set the type of variance estimate."""
        if self.fitted:
            raise RuntimeError("'var_type' cannot be set after fitting.")
        elif var_type in self._var_types:
            self._var_type = var_type
        else:
            raise ValueError(f"Invalid value for 'var_type': {var_type}.")

    def __init__(self, conf_type="log-log", conf_level=0.95,
                 var_type="greenwood"):
        """Initialize the Kaplan-Meier survival function estimator.

        Parameters
        ----------
        conf_type : str, optional (default: "log-log")
            Type of confidence interval for the survival function estimate to
            report. Accepted values:
                * "linear"
                * "log"
                * "log-log"
                * "logit"
                * "arcsin"
            See this class's docstring for details.
        conf_level : float, optional (default: 0.95)
            Confidence level of the confidence intervals.
        var_type : str, optional (default: "greenwood")
            Type of variance estimate for the survival function to compute.
            Accepted values:
                * "greenwood"
                * "aalen-johansen"
            See this class's docstring for details.
        """
        # Parameter validation is done in each parameter's setter method
        self.conf_type = conf_type
        self.conf_level = conf_level
        self.var_type = var_type

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

            # Sum occurring in the Greenwood and Aalen-Johansen variance
            # estimates
            if self._var_type == "greenwood":
                # Greenwood's formula
                with np.errstate(divide="ignore"):
                    var_sum = np.cumsum(e / r / (r - e))
            elif self._var_type == "aalen-johansen":
                # Aalen-Johansen estimate
                var_sum = np.cumsum(e / (r ** 2))
            else:
                # This should not be reachable
                raise RuntimeError(f"Invalid variance type: {self._var_type}.")

            # Survival function variance estimates
            with np.errstate(invalid="ignore"):
                self._survival_var[i] = (self._survival[i] ** 2) * var_sum

            # Standard normal quantile for confidence intervals
            z = st.norm.ppf((1 - self.conf_level) / 2)

            # Compute confidence intervals at the observed event times
            if self._conf_type == "linear":
                # Normal approximation CI
                c = z * np.sqrt(self._survival_var[i])
                lower = self._survival[i] + c
                upper = self._survival[i] - c
            elif self._conf_type == "log":
                # CI based on a delta method CI for log(S(t))
                with np.errstate(invalid="ignore"):
                    c = z * np.sqrt(var_sum)
                    lower = self._survival[i] * np.exp(c)
                    upper = self._survival[i] * np.exp(-c)
            elif self._conf_type == "log-log":
                # CI based on a delta method CI for -log(-log(S(t)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    c = z * np.sqrt(var_sum) / np.log(self._survival[i])
                    lower = self._survival[i] ** np.exp(c)
                    upper = self._survival[i] ** np.exp(-c)
            elif self._conf_type == "logit":
                # CI based on a delta method CI for log(S(t)/(1-S(t)))
                with np.errstate(invalid="ignore"):
                    odds = self._survival[i] / (1 - self._survival[i])
                    c = z * np.sqrt(var_sum) / (1 - self._survival[i])
                    lower = 1 - 1 / (1 + odds * np.exp(c))
                    upper = 1 - 1 / (1 + odds * np.exp(-c))
                pass
            elif self._conf_type == "arcsin":
                # CI based on a delta method CI for arcsin(sqrt(S(t))
                with np.errstate(invalid="ignore"):
                    arcsin = np.arcsin(np.sqrt(self._survival[i]))
                    odds = self._survival[i] / (1 - self._survival[i])
                    c = 0.5 * z * np.sqrt(odds * var_sum)
                    lower = np.sin(np.maximum(0., arcsin + c)) ** 2
                    upper = np.sin(np.minimum(np.pi / 2, arcsin - c)) ** 2
            else:
                # This should not be reachable
                raise RuntimeError(
                    f"Invalid confidence interval type: {self._conf_type}.")

            # Force confidence interval bounds to be between 0 and 1
            with np.errstate(invalid="ignore"):
                self._survival_ci_lower[i] = np.maximum(lower, 0.)
                self._survival_ci_upper[i] = np.minimum(upper, 1.)

        self.fitted = True
        return self
