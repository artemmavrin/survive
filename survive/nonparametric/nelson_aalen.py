"""Implements the Nelson-Aalen cumulative hazard estimator."""

import numpy as np
import scipy.stats as st

from .base import NonparametricEstimator
from ..survival_data import SurvivalData


class NelsonAalen(NonparametricEstimator):
    r"""Nelson-Aalen nonparametric cumulative hazard estimator.

    This estimator was suggested by Nelson in [1]_ in the context of
    reliability, and it was rediscovered and generalized by Aalen in [2]_.

    Parameters
    ----------
    conf_type : {'log', 'linear'}
        Type of confidence interval for the cumulative hazard estimate to
        report.

    conf_level : float
        Confidence level of the confidence intervals.

    var_type : {'aalen', 'greenwood'}
        Type of variance estimate to compute.

    tie_break : {'discrete', 'continuous'}
        Specify how to handle tied event times.

    Notes
    -----
    Suppose we have observed right-censored and left-truncated event times. Let
    :math:`T_1 < T_2 < \cdots` denote the ordered distinct event times. Let
    :math:`N(t)` be the number of events observed up to time :math:`t`,
    let :math:`Y(t)` denote the number of individuals at risk (under observation
    but not yet censored or "dead") at time :math:`t`, and let

    .. math::
        J(t) = \begin{cases} 1 & \text{if $Y(t) > 0$,} \\
        0 & \text{otherwise.} \end{cases}

    The *Nelson-Aalen estimator* estimates the cumulative hazard function of the
    time-to-event distribution by

    .. math::
        \widehat{A}(t) = \int_0^t \frac{J(s)}{Y(s)} \, dN(s).

    This formula, proposed in [2]_, is computed as a sum in one of two ways
    depending on how tied event times are handled (cf. Section 3.1.3 in [3]_).
    This is governed by the `tie_break` parameter.

    * If `tie_break` is "discrete", then it is assumed that tied events are
      possible, and we compute the integral defining the Nelson-Aalen estimator
      directly, leading to

      .. math::

        \widehat{A}(t) = \sum_{j : T_j \leq t} \frac{\Delta N(T_j)}{Y(T_j)}.

      Here :math:`\Delta N(T_j)` is the number of events occurring at time
      :math:`T_j`.

    * If `tie_break` is "continuous", then it is assumed that tied events only
      happen due to grouping or rounding, and the tied times are treated as if
      they happened in succession, each one immediately following the previous
      one. This leads to the estimator

      .. math::

        \widehat{A}(t) = \sum_{j : T_j \leq t}
        \sum_{k=0}^{\Delta N(T_j) - 1} \frac{1}{Y(T_j) - k}.

    The variance of the Nelson-Aalen estimator is estimated by one of two
    estimators suggested by [4]_. You can select the variance estimator by using
    the `var_type` parameter.

    * If `var_type` is "aalen", then the variance estimator derived in [2]_ is
      used:

      .. math::

        \widehat{\mathrm{Var}}(\widehat{A}(t))
        = \int_0^t \frac{J(s)}{Y(s)^2} \, dN(s).

      This integral is computed in one of two ways depending on `tie_break`:

      * If `tie_break` is "discrete", then the variance estimator is computed
        as

        .. math::

          \widehat{\mathrm{Var}}(\widehat{A}(t))
          = \sum_{j : T_j \leq t} \frac{\Delta N(T_j)}{Y(T_j)^2}.

      * If `tie_break` is "continuous", then the variance estimator is computed
        as

        .. math::

          \widehat{\mathrm{Var}}(\widehat{A}(t))
          = \sum_{j : T_j \leq t} \sum_{k=0}^{\Delta N(T_j) - 1}
          \frac{1}{\left(Y(T_j) - k\right)^2}.

      This estimator of the variance was found to generally overestimate the
      true variance of the Nelson-Aalen estimator [4]_.

    * If `var_type` is "greenwood", then the Greenwood-type estimator derived in
      [4]_ is used:

      .. math::

        \widehat{\mathrm{Var}}(\widehat{A}(t))
        &= \int_0^t \frac{J(s) (Y(s) - \Delta N(s))}{Y(s)^3} \, dN(s) \\
        &= \sum_{j : T_j \leq t}
        \frac{(Y(T_j) - \Delta N(T_j)) \Delta N(T_j)}{Y(T_j)^3}.

      This estimator tends to have a uniformly lower mean squared error than the
      Aalen estimator, but it also tends to underestimate the true variance of
      the Nelson-Aalen estimator [4]_.

    The difference between these two variance estimators is only significant at
    times when the risk set is small. Klein [4]_ recommends the Aalen estimator
    over the Greenwood-type estimator.

    The two types of confidence intervals ("log" and "linear") provided here are
    presented in [5]_. They are based on the asymptotic normality of the
    Nelson-Aalen estimator and are derived from the delta method by suitable
    transformations of the estimator. The "log" intervals are more accurate for
    smaller sample sizes, but both methods are equivalent for large samples
    [5]_.

    References
    ----------
    .. [1] Wayne Nelson. "Theory and Applications of Hazard Plotting for
        Censored Failure Data". Technometrics, Volume 14, Number 4 (1972),
        pp. 945--966. `JSTOR <http://www.jstor.org/stable/1267144>`__.
    .. [2] Odd Aalen. "Nonparametric Inference for a Family of Counting
        Processes". The Annals of Statistics, Volume 6, Number 4 (1978),
        pp. 701--726. `JSTOR <http://www.jstor.org/stable/2958850>`__.
    .. [3] Odd O. Aalen, Ørnulf Borgan, and Håkon K. Gjessing. Survival and
        Event History Analysis. A Process Point of View. Springer-Verlag, New
        York (2008) pp. xviii+540.
        `DOI <https://doi.org/10.1007/978-0-387-68560-1>`__.
    .. [4] John P. Klein. "Small sample moments of some estimators of the
        variance of the Kaplan-Meier and Nelson-Aalen estimators." Scandinavian
        Journal of Statistics. Volume 18, Number 4 (1991), pp. 333--40.
        `JSTOR <http://www.jstor.org/stable/4616215>`__.
    .. [5] Ole Bie, Ørnulf Borgan, and Knut Liestøl. "Confidence Intervals and
        Confidence Bands for the Cumulative Hazard Rate Function and Their Small
        Sample Properties." Scandinavian Journal of Statistics, Volume 14,
        Number 3 (1987), pp. 221--33.
        `JSTOR <http://www.jstor.org/stable/4616065>`__.
    """
    model_type = "Nelson-Aalen estimator"
    _estimand = "cumulative hazard"
    _estimate0 = 0.

    _conf_types = ("linear", "log")
    _var_types = ("aalen", "greenwood")
    _tie_breaks = ("continuous", "discrete")

    def __init__(self, *, conf_type="log", conf_level=0.95,
                 var_type="aalen", tie_break="discrete"):
        self.conf_type = conf_type
        self.conf_level = conf_level
        self.var_type = var_type
        self.tie_break = tie_break

    def fit(self, time, **kwargs):
        """Fit the Nelson-Aalen estimator to survival data.

        Parameters
        ----------
        time : one-dimensional array-like or str or SurvivalData
            The observed times, or all the survival data. If this is a
            :class:`survive.SurvivalData` instance, then it is used to fit the
            estimator and any other parameters are ignored. Otherwise, `time`
            and the keyword arguments in `kwargs` are used to initialize a
            :class:`survive.SurvivalData` object on which this estimator is
            fitted.

        **kwargs : keyword arguments
            Any additional keyword arguments used to initialize a
            :class:`survive.SurvivalData` instance.

        Returns
        -------
        survive.nonparametric.NelsonAalen
            This estimator.

        See Also
        --------
        survive.SurvivalData : Structure used to store survival data.
        """
        if isinstance(time, SurvivalData):
            self._data = time
        else:
            self._data = SurvivalData(time, **kwargs)

        # Compute the Nelson-Aalen estimator and related quantities at the
        # distinct failure times within each group
        self.estimate_ = []
        self.estimate_var_ = []
        self.estimate_ci_lower_ = []
        self.estimate_ci_upper_ = []
        for i, group in enumerate(self._data.group_labels):
            # d = number of events at an event time, y = size of the risk set at
            # an event time
            d = self._data.events[group].n_events
            y = self._data.events[group].n_at_risk

            # Compute the Nelson-Aalen estimator increments
            if self._tie_break == "discrete":
                na_inc = d / y
            elif self._tie_break == "continuous":
                na_inc = np.empty(len(d), dtype=np.float_)
                for j in range(len(d)):
                    na_inc[j] = np.sum(1 / (y[j] - np.arange(d[j])))
            else:
                # This should not be reachable
                raise RuntimeError(
                    f"Invalid tie-breaking scheme: {self._tie_break}.")

            # Compute the variance estimate increments
            if self._var_type == "greenwood":
                var_inc = (y - d) * d / (y ** 3)
            elif self.var_type == "aalen":
                if self._tie_break == "discrete":
                    var_inc = d / (y ** 2)
                elif self._tie_break == "continuous":
                    var_inc = np.empty(len(d), dtype=np.float_)
                    for j in range(len(d)):
                        var_inc[j] = np.sum(1 / (y[j] - np.arange(d[j])) ** 2)
                else:
                    # This should not be reachable
                    raise RuntimeError(
                        f"Invalid tie-breaking scheme: {self._tie_break}.")
            else:
                # This should not be reachable
                raise RuntimeError(
                    f"Invalid variance type: {self._var_type}.")

            # Compute Nelson-Aalen estimate and variance estimates
            self.estimate_.append(np.cumsum(na_inc))
            self.estimate_var_.append(np.cumsum(var_inc))

            # Standard normal quantile for confidence intervals
            z = st.norm.ppf((1 - self.conf_level) / 2)

            # Compute confidence intervals at the observed event times
            if self._conf_type == "linear":
                # Normal approximation CI
                c = z * np.sqrt(self.estimate_var_[i])
                lower = self.estimate_[i] + c
                upper = self.estimate_[i] - c
            elif self.conf_type == "log":
                se = np.sqrt(self.estimate_var_[i])
                estimate = self.estimate_[i]
                a = np.exp(z * se / estimate)
                lower = estimate * a
                upper = estimate / a
            else:
                # This should not be reachable
                raise RuntimeError(
                    f"Invalid confidence interval type: {self._conf_type}.")

            # Force confidence interval lower bound to be 0
            with np.errstate(invalid="ignore"):
                self.estimate_ci_lower_.append(np.maximum(lower, 0.))
                self.estimate_ci_upper_.append(upper)

        self.fitted = True
        return self
