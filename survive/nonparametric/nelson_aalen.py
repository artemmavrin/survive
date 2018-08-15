"""Implements the Nelson-Aalen cumulative hazard estimator."""

import numpy as np
import scipy.stats as st

from .base import NonparametricEstimator, NonparametricSurvival
from ..survival_data import SurvivalData

_NELSON_AALEN_CONF_TYPES = ("linear", "log")
_NELSON_AALEN_VAR_TYPES = ("aalen", "greenwood")
_NELSON_AALEN_TIE_BREAKS = ("continuous", "discrete")


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

    _conf_types = _NELSON_AALEN_CONF_TYPES
    _var_types = _NELSON_AALEN_VAR_TYPES
    _tie_breaks = _NELSON_AALEN_TIE_BREAKS

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
        self.estimate_ = dict()
        self.var_ = dict()
        self.ci_lower_ = dict()
        self.ci_upper_ = dict()
        for group in self._data.group_labels:
            n_events = self._data.events[group].n_events
            n_at_risk = self._data.events[group].n_at_risk

            # Compute the Nelson-Aalen estimator
            estimate = _nelson_aalen_fit(n_events=n_events, n_at_risk=n_at_risk,
                                         tie_break=self._tie_break)
            self.estimate_[group] = estimate

            # Estimate the variance of the Nelson-Aalen estimator
            variance = _nelson_aalen_var(n_events=n_events, n_at_risk=n_at_risk,
                                         var_type=self._var_type,
                                         tie_break=self._tie_break)
            self.var_[group] = variance

            # Construct confidence intervals at the distinct event times
            self.ci_lower_[group], self.ci_upper_[group] = \
                _nelson_aalen_ci(estimate=estimate, variance=variance,
                                 conf_type=self._conf_type,
                                 conf_level=self._conf_level)

        self.fitted = True
        return self


def _nelson_aalen_fit(n_events, n_at_risk, tie_break):
    """Compute the Nelson-Aalen estimator."""
    # Compute the Nelson-Aalen estimator increments
    if tie_break == "discrete":
        increments = n_events / n_at_risk
    elif tie_break == "continuous":
        k = n_events.shape[0]
        increments = np.empty(k, dtype=np.float_)
        for j in range(k):
            increments[j] = np.sum(1 / (n_at_risk[j] - np.arange(n_events[j])))
    else:
        # This should not be reachable
        raise RuntimeError(f"Invalid tie-breaking scheme: {tie_break}.")

    return np.cumsum(increments)


def _nelson_aalen_var(n_events, n_at_risk, var_type, tie_break):
    """Estimate the variance of the Nelson-Aalen estimator."""
    if var_type == "greenwood":
        increments = (n_at_risk - n_events) * n_events / (n_at_risk ** 3)
    elif var_type == "aalen":
        if tie_break == "discrete":
            increments = n_events / (n_at_risk ** 2)
        elif tie_break == "continuous":
            k = n_events.shape[0]
            increments = np.empty(k, dtype=np.float_)
            for j in range(k):
                denominator = (n_at_risk[j] - np.arange(n_events[j])) ** 2
                increments[j] = np.sum(1 / denominator)
        else:
            # This should not be reachable
            raise RuntimeError(f"Invalid tie-breaking scheme: {tie_break}.")
    else:
        # This should not be reachable
        raise RuntimeError(f"Invalid variance type: {var_type}.")

    return np.cumsum(increments)


def _nelson_aalen_ci(estimate, variance, conf_type, conf_level):
    """Construct confidence intervals for the Nelson-Aalen estimates."""
    # Standard normal quantile for normal approximation confidence intervals
    quantile = st.norm.ppf((1 - conf_level) / 2)

    # Compute confidence intervals at the observed event times
    if conf_type == "linear":
        error = quantile * np.sqrt(variance)
        lower = estimate + error
        upper = estimate - error
    elif conf_type == "log":
        error = np.exp(quantile * np.sqrt(variance) / estimate)
        lower = estimate * error
        upper = estimate / error
    else:
        # This should not be reachable
        raise RuntimeError(f"Invalid confidence interval type: {conf_type}.")

    # Force confidence interval lower bound to be 0
    lower = np.maximum(lower, 0.)

    return lower, upper


class Breslow(NonparametricSurvival):
    r"""Breslow nonparametric survival function estimator.

    Parameters
    ----------
    conf_type : {'log', 'linear'}
        Type of confidence interval to report.

    conf_level : float
        Confidence level of the confidence intervals.

    var_type : {'aalen', 'greenwood'}
        Type of variance estimate to compute.

    tie_break : {'discrete', 'continuous'}
        Specify how to handle tied event times.

    See Also
    --------
    survive.NelsonAalen : Nelson-Aalen cumulative hazard function estimator.

    Notes
    -----
    The *Breslow estimator* is a nonparametric estimator of the survival
    function of a time-to-event distribution defined as the exponential of the
    negative of the Nelson-Aalen cumulative hazard function estimator
    :math:`\widehat{A}(t)`:

    .. math::

        \widehat{S}(t) = \exp(-\widehat{A}(t)).

    This estimator was introduced in a discussion [1]_ following [2]_. It was
    later studied by Fleming and Harrington in [3]_, and it is sometimes called
    the *Fleming-Harrington estimator*.

    The parameters of this class are identical to the parameters of
    :class:`survive.NelsonAalen`. The Breslow survival function estimates and
    confidence interval bounds are transformations of the Nelson-Aalen
    cumulative hazard estimates and confidence interval bounds, respectively.
    The variance estimate for the Breslow estimator is computed using the
    variance estimate for the Nelson-Aalen estimator using the Nelson-Aalen
    estimator's asymptotic normality and the delta method:

    .. math::

        \widehat{\mathrm{Var}}(\widehat{S}(t))
        = \widehat{S}(t)^2 \widehat{\mathrm{Var}}(\widehat{A}(t))

    Comparisons of the Breslow estimator and the more popular Kaplan-Meier
    estimator (cf. :class:`survive.KaplanMeier`) can be found in [3]_ and [4]_.
    One takeaway is that the Breslow estimator was found to be more biased than
    the Kaplan-Meier estimator, but the Breslow estimator had a lower mean
    squared error.

    References
    ----------
    .. [1] N. E. Breslow. "Discussion of Professor Cox’s Paper". Journal of the
        Royal Statistical Society. Series B (Methodological), Volume 34,
        Number 2 (1972), pp. 216--217.
    .. [2] D. R. Cox. "Regression Models and Life-Tables". Journal of the Royal
        Statistical Society. Series B (Methodological), Volume 34, Number 2
        (1972), pp. 187--202. `JSTOR <http://www.jstor.org/stable/2985181>`__.
    .. [3] Thomas R. Fleming and David P. Harrington. "Nonparametric Estimation
        of the Survival Distribution in Censored Data". Communications in
        Statistics - Theory and Methods, Volume 13, Number 20 (1984),
        pp. 2469--2486. `DOI <https://doi.org/10.1080/03610928408828837>`__.
    .. [4] Xuelin Huang and Robert L. Strawderman. "A Note on the Breslow
        Survival Estimator". Journal of Nonparametric Statistics, Volume 18,
        Number 1 (2006), pp. 45--56.
        `DOI <https://doi.org/10.1080/10485250500491661>`__.
    """
    model_type = "Breslow estimator"

    _conf_types = _NELSON_AALEN_CONF_TYPES
    _var_types = _NELSON_AALEN_VAR_TYPES
    _tie_breaks = _NELSON_AALEN_TIE_BREAKS

    def __init__(self, *, conf_type="log", conf_level=0.95, var_type="aalen",
                 tie_break="discrete"):
        self.conf_type = conf_type
        self.conf_level = conf_level
        self.var_type = var_type
        self.tie_break = tie_break

    def fit(self, time, **kwargs):
        """Fit the Breslow estimator to survival data.

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
        survive.NelsonAalen : Nelson-Aalen cumulative hazard estimator.
        """

        nelson_aalen = NelsonAalen(conf_type=self.conf_type,
                                   conf_level=self.conf_level,
                                   var_type=self.var_type,
                                   tie_break=self.tie_break)
        nelson_aalen.fit(time, **kwargs)

        self._data = nelson_aalen.data_

        self.estimate_ = dict()
        self.var_ = dict()
        self.ci_lower_ = dict()
        self.ci_upper_ = dict()
        for group in self._data.group_labels:
            # Extract Nelson-Aalen estimates for the current group
            na_estimate = nelson_aalen.estimate_[group]
            na_var = nelson_aalen.var_[group]
            na_ci_lower = nelson_aalen.ci_lower_[group]
            na_ci_upper = nelson_aalen.ci_upper_[group]

            # The Breslow estimator is the exponential of the negative of the
            # Nelson-Aalen estimator
            survival = np.exp(-na_estimate)
            self.estimate_[group] = survival

            # Estimate the Breslow estimator variance using the delta method
            self.var_[group] = (survival ** 2) * na_var

            # Get Breslow estimator confidence intervals by transforming the
            # Nelson-Aalen estimator confidence intervals
            self.ci_lower_[group] = np.exp(-na_ci_upper)
            self.ci_upper_[group] = np.exp(-na_ci_lower)

        self.fitted = True
        return self
