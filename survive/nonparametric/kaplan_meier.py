"""The Kaplan-Meier nonparametric survival function estimator."""

import numpy as np
import scipy.stats as st

from .base import NonparametricSurvival
from ..survival_data import SurvivalData
from ..utils.validation import check_int


class KaplanMeier(NonparametricSurvival):
    r"""Kaplan-Meier nonparametric survival function estimator.

    The Kaplan-Meier estimator [1]_ is also called the product-limit estimator.
    Much of this implementation is inspired by the R package ``survival`` [2]_.

    For a quick introduction to the Kaplan-Meier estimator, see e.g. Section 4.2
    in [3]_ or Section 1.4.1 in [4]_. For a more thorough treatment, see Chapter
    4 in [5]_.

    Parameters
    ----------
    conf_type : {'log-log', 'linear', 'log', 'logit', 'arcsin'}
        Type of confidence interval for the survival function estimate to
        report.

    conf_level : float
        Confidence level of the confidence intervals.

    var_type : {'greenwood', 'aalen-johansen', 'bootstrap'}
        Type of variance estimate for the survival function to compute.

    tie_break : {'discrete', 'continuous'}
        Specify how to handle tied event times when computing the Aalen-Johansen
        variance estimate (when `var_type` is "aalen-johansen"). Ignored for
        other values of `var_type`.

    n_boot : int, optional
        Number of bootstrap samples to draw when estimating the survival
        function variance using the bootstrap (when `var_type` is "bootstrap").
        Ignored for other values of `var_type`.

    random_state : int or numpy.random.RandomState, optional
        Random number generator (or a seed for one) used for sampling and for
        variance computations if `var_type` is "bootstrap". Ignored for other
        values of `var_type`.

    Notes
    -----
    Suppose we have observed right-censored and left-truncated event times. Let
    :math:`T_1 < T_2 < \cdots` denote the ordered distinct event times. Let
    :math:`\Delta N(T_j)` be the number of events observed at time :math:`T_j`,
    and let :math:`Y(T_j)` denote the number of individuals at risk (under
    observation but not yet censored or "dead") at time :math:`T_j`. The
    *Kaplan-Meier estimator* estimates the survival function :math:`S(t)` of the
    time-to-event distribution by

    .. math::

        \widehat{S}(t)
        = \prod_{j : T_j \leq t} \left(1 - \frac{\Delta N(T_j)}{Y(T_j)}\right).

    There are several supported ways of estimating the Kaplan-Meier variance
    :math:`\mathrm{Var}(\widehat{S}(t))`, each one corresponding to a different
    value of `var_type`:

    "greenwood"
        This is the classical Greenwood's formula [6]_:

        .. math::

            \widehat{\mathrm{Var}}(\widehat{S}(t))
            = \widehat{S}(t)^2 \sum_{j : T_j \leq t}
            \frac{\Delta N(T_j)}{Y(T_j) (Y(T_j) - \Delta N(T_j))}.

    "aalen-johansen"
        This uses the Poisson moment approximation to the binomial suggested in
        [7]_. This method requires choosing how to handle tied event times by
        specifying the parameter `tie_break`. See Sections 3.1.3 and 3.2.2 in
        [8]_. Possible values are

        "discrete"
            Tied event times are possible and are treated as simultaneous. The
            variance estimate is

            .. math::

                \widehat{\mathrm{Var}}(\widehat{S}(t))
                = \widehat{S}(t)^2 \sum_{j : T_j \leq t}
                \frac{\Delta N(T_j)}{Y(T_j)^2}.

        "continuous"
            True event times almost surely don't coincide, and any observed ties
            are due to grouping or rounding. Tied event times will be treated as
            if each one occurred in succession, each one immediately following
            the previous one. The variance estimate is

            .. math::

                \widehat{\mathrm{Var}}(\widehat{S}(t))
                = \widehat{S}(t)^2 \sum_{j : T_j \leq t}
                \sum_{k=0}^{\Delta N(T_j) - 1}
                \frac{1}{\left(Y(T_j) - k\right)^2}.

        This method is less frequently used than Greenwood's formula, and the
        two methods are usually close to each other numerically. However, [9]_
        recommends using Greenwood's formula because it is less biased and has
        comparable or lower mean squared error.

    "bootstrap"
        This uses the bootstrap to estimate the survival function variance
        [10]_. Specifically, one chooses a positive integer :math:`B` (the
        number of bootstrap samples `n_boot`), forms :math:`B` bootstrap samples
        by sampling with replacement from the data, and computes the
        Kaplan-Meier estimate :math:`\widehat{S}_b^*(t)` for each time :math:`t`
        and each :math:`b=1,\ldots,B`. The resulting variance estimate is

        .. math::

            \widehat{\mathrm{Var}}(\widehat{S}(t))
            = \frac{1}{B} \sum_{b=1}^B \left(\widehat{S}_b^*(t)
            - \frac{1}{B} \sum_{c=1}^B \widehat{S}_c^*(t)\right)^2

    Having chosen a variance estimate, we can estimate the standard error by

    .. math ::

        \widehat{\mathrm{SE}}(\widehat{S}(t))
        = \sqrt{\widehat{\mathrm{Var}}(\widehat{S}(t))}.

    Confidence intervals for :math:`p=\widehat{S}(t)` exploit the asymptotic
    normality of the Kaplan-Meier estimator by choosing a strictly increasing
    and differentiable transformation :math:`f(p)` and applying the delta
    method, which which states that :math:`f(p)` is also asymptotically normal,
    with standard error :math:`\mathrm{SE}(p) f^\prime(p)`. Consequently, a
    normal approximation confidence interval for :math:`f(p)` is

    .. math::

        f(p) \pm z \widehat{\mathrm{SE}}(p) f^\prime(p)

    where :math:`z` is the (1-`conf_level`)/2-quantile of the standard normal
    distribution. A confidence interval for :math:`p` is then

    .. math::

        f^{-1}\left(f(p) \pm z \widehat{\mathrm{SE}}(p) f^\prime(p)\right)

    These general types of confidence intervals were proposed in [11]_. Our
    implementation also shrinks the intervals to be between 0 and 1 if
    necessary. We list the supported transformations below.

        =========== ===========================
        `conf_type` :math:`f(p)`
        =========== ===========================
        "linear"    :math:`p`
        "log"       :math:`\log(p)`
        "log-log"   :math:`-\log(-\log(p))`
        "logit"     :math:`\log(p/(1-p))`
        "arcsin"    :math:`\arcsin(\sqrt{p})`
        =========== ===========================

    The confidence intervals implemented here are equivalent for large samples
    (i.e., asymptotically). For small samples (as small as 25 observations with
    up to 50% censoring away from the right tail), the "log" and "arcsin"
    confidence intervals have been shown to give close to the correct coverage
    probability, whereas the "linear" confidence interval needs much larger
    sample sizes to perform similarly [11]_. For small samples, the "arcsin"
    intervals tend to be conservative, the "log" intervals tend to be
    slightly liberal, and the "linear" intervals tend to be very liberal [11]_.

    The "log" intervals were introduced in the first edition of [4]_, and the
    "arcsin" intervals were introduced in [12]_.

    References
    ----------
    .. [1] E. L. Kaplan and P. Meier. "Nonparametric estimation from incomplete
        observations". Journal of the American Statistical Association, Volume
        53, Issue 282 (1958), pp. 457--481.
        `DOI <https://doi.org/10.2307/2281868>`__.
    .. [2] Terry M. Therneau. A Package for Survival Analysis in S. version 2.38
        (2015). `CRAN <https://CRAN.R-project.org/package=survival>`__.
    .. [3] D. R. Cox and D. Oakes. Analysis of Survival Data. Chapman & Hall,
        London (1984), pp. ix+201.
    .. [4] John D. Kalbfleisch and Ross L. Prentice. The Statistical Analysis of
        Failure Time Data. Second Edition. Wiley (2002) pp. xiv+439.
    .. [5] John P. Klein and Melvin L. Moeschberger. Survival Analysis.
        Techniques for Censored and Truncated Data. Second Edition.
        Springer-Verlag, New York (2003) pp. xvi+538.
        `DOI <https://doi.org/10.1007/b97377>`__.
    .. [6] M. Greenwood. "The natural duration of cancer". Reports on Public
        Health and Medical Subjects. Volume 33 (1926), pp. 1--26.
    .. [7] Odd O. Aalen and Søren Johansen. "An empirical transition matrix for
        non-homogeneous Markov chains based on censored observations."
        Scandinavian Journal of Statistics. Volume 5, Number 3 (1978),
        pp. 141--150. `JSTOR <http://www.jstor.org/stable/4615704>`__.
    .. [8] Odd O. Aalen, Ørnulf Borgan, and Håkon K. Gjessing. Survival and
        Event History Analysis. A Process Point of View. Springer-Verlag, New
        York (2008) pp. xviii+540.
        `DOI <https://doi.org/10.1007/978-0-387-68560-1>`__.
    .. [9] John P. Klein. "Small sample moments of some estimators of the
        variance of the Kaplan-Meier and Nelson-Aalen estimators." Scandinavian
        Journal of Statistics. Volume 18, Number 4 (1991), pp. 333--40.
        `JSTOR <http://www.jstor.org/stable/4616215>`__.
    .. [10] Bradley Efron. "Censored data and the bootstrap." Journal of the
        American Statistical Association. Volume 76, Number 374 (1981),
        pp. 312--19. `DOI <https://doi.org/10.2307/2287832>`__.
    .. [11] Ørnulf Borgan and Knut Liestøl. "A note on confidence intervals and
        bands for the survival function based on transformations." Scandinavian
        Journal of Statistics. Volume 17, Number 1 (1990), pp. 35--41.
        `JSTOR <http://www.jstor.org/stable/4616153>`__.
    .. [12] Vijayan N. Nair.  "Confidence Bands for Survival Functions with
        Censored Data: A Comparative Study." Technometrics, Volume 26, Number 3,
        (1984), pp. 265--75. `DOI <https://doi.org/10.2307/1267553>`__.
    """
    model_type = "Kaplan-Meier estimator"

    _conf_types = ("arcsin", "linear", "log", "log-log", "logit")
    _var_types = ("aalen-johansen", "bootstrap", "greenwood")
    _tie_breaks = ("continuous", "discrete")

    # Number of bootstrap samples to draw
    _n_boot: int

    @property
    def n_boot(self):
        """Number of bootstrap samples to draw when `var_type` is "bootstrap".
        Not used for any other values of `var_type`.
        """
        return self._n_boot

    @n_boot.setter
    def n_boot(self, n_boot):
        """Set the number of bootstrap samples to draw for bootstrap variance
        estimates.
        """
        self._n_boot = check_int(n_boot, minimum=1)

    def __init__(self, *, conf_type="log-log", conf_level=0.95,
                 var_type="greenwood", tie_break="discrete", n_boot=500,
                 random_state=None):
        # Parameter validation is done in each parameter's setter method
        self.conf_type = conf_type
        self.conf_level = conf_level
        self.var_type = var_type
        self.tie_break = tie_break
        self.n_boot = n_boot
        self.random_state = random_state

    def fit(self, time, **kwargs):
        """Fit the Kaplan-Meier estimator to survival data.

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
        survive.nonparametric.KaplanMeier
            This estimator.

        See Also
        --------
        survive.SurvivalData : Structure used to store survival data.
        """
        if isinstance(time, SurvivalData):
            self._data = time
        else:
            self._data = SurvivalData(time, **kwargs)

        # Compute the Kaplan-Meier product-limit estimator and related
        # quantities at the distinct failure times within each group
        self.estimate_ = dict()
        self.var_ = dict()
        self.ci_lower_ = dict()
        self.ci_upper_ = dict()
        for group in self._data.group_labels:
            n_events = self._data.events[group].n_events
            n_at_risk = self._data.events[group].n_at_risk

            # Compute the Kaplan-Meier estimator at the event times
            survival = _km_fit(n_events=n_events, n_at_risk=n_at_risk)
            self.estimate_[group] = survival

            # `variance` is the variance of the Kaplan-Meier estimator, `sigma2`
            # is the variance of a related cumulative hazard estimator that
            # might be used for confidence interval computations later
            variance, sigma2 = _km_var(survival=survival, n_events=n_events,
                                       n_at_risk=n_at_risk, data=self._data,
                                       group=group,
                                       random_state=self._random_state,
                                       n_boot=self._n_boot,
                                       var_type=self._var_type,
                                       tie_break=self._tie_break)
            self.var_[group] = variance

            self.ci_lower_[group], self.ci_upper_[group] = \
                _km_ci(survival=survival, variance=variance, sigma2=sigma2,
                       conf_type=self._conf_type, conf_level=self._conf_level)

            # Make sure that variance estimates and confidence intervals are NaN
            # when the estimated survival probability is zero
            mask = (survival == 0.)
            self.var_[group][mask] = np.nan
            self.ci_lower_[group][mask] = np.nan
            self.ci_upper_[group][mask] = np.nan

        self.fitted = True
        return self


def _km_fit(n_events, n_at_risk):
    """Compute the Kaplan-Meier estimator."""
    return np.cumprod(1. - n_events / n_at_risk)


def _km_var(survival, n_events, n_at_risk, data: SurvivalData, group,
            random_state, n_boot, var_type, tie_break):
    """Compute the variance of the Kaplan-Meier estimator."""
    if var_type == "greenwood":
        # Greenwood's formula
        variance, sigma2 = _km_var_greenwood(survival=survival,
                                             n_events=n_events,
                                             n_at_risk=n_at_risk)
    elif var_type == "aalen-johansen":
        # Aalen-Johansen variance estimate
        variance, sigma2 = _km_var_aalen_johansen(survival=survival,
                                                  n_events=n_events,
                                                  n_at_risk=n_at_risk,
                                                  tie_break=tie_break)
    elif var_type == "bootstrap":
        # Estimate the survival function variance using the bootstrap
        variance, sigma2 = _km_var_bootstrap(survival=survival, data=data,
                                             group=group,
                                             random_state=random_state,
                                             n_boot=n_boot)
    else:
        # This should not be reachable
        raise RuntimeError(f"Invalid variance type: {var_type}.")

    return variance, sigma2


def _km_var_greenwood(survival, n_events, n_at_risk):
    """Estimate the Kaplan-Meier variance using Greenwood's formula."""
    with np.errstate(divide="ignore"):
        sigma2 = np.cumsum(n_events / n_at_risk / (n_at_risk - n_events))
        variance = (survival ** 2) * sigma2
    return variance, sigma2


def _km_var_aalen_johansen(survival, n_events, n_at_risk, tie_break):
    """Estimate the Kaplan-Meier variance using the Aalen-Johansen formula."""
    if tie_break == "discrete":
        sigma2 = np.cumsum(n_events / (n_at_risk ** 2))
    elif tie_break == "continuous":
        # Increments of sum in equation (3.14) on page 84 of Aalen, Borgan, and
        # Gjessing (2008)
        k = len(n_events)
        inc = np.empty(k, dtype=np.float_)
        for j in range(k):
            inc[j] = np.sum(1 / (n_at_risk[j] - np.arange(n_events[j])) ** 2)
        sigma2 = np.cumsum(inc)
    else:
        # This should not be reachable
        raise RuntimeError(f"Invalid tie-breaking scheme: {tie_break}.")

    with np.errstate(divide="ignore"):
        variance = (survival ** 2) * sigma2

    return variance, sigma2


def _km_var_bootstrap(survival, data: SurvivalData, group, random_state,
                      n_boot):
    """Estimate Kaplan-Meier survival function variance using the bootstrap."""
    # Extract observed times, censoring indicators, and entry times for the
    # specified group
    mask = (data.group == group)
    time = np.asarray(data.time[mask])
    status = np.asarray(data.status[mask])
    entry = np.asarray(data.entry[mask])

    sample_size = time.shape[0]

    # Distinct true event times
    event_times = np.asarray(data.events[group].time)
    n_event_times = event_times.shape[0]

    # Initialize array of bootstrap Kaplan-Meier survival function estimates at
    # the observed true event times
    survival_boot = np.empty(shape=(n_boot, n_event_times), dtype=np.float_)

    # The bootstrap
    for i in range(n_boot):
        # Draw a bootstrap sample
        ind_b = random_state.choice(sample_size, size=sample_size, replace=True)
        time_b = time[ind_b]
        status_b = status[ind_b]
        entry_b = entry[ind_b]

        n_events = np.empty(shape=(n_event_times,), dtype=np.int_)
        n_at_risk = np.empty(shape=(n_event_times,), dtype=np.int_)
        for j, e_time in enumerate(event_times):
            n_events[j] = np.sum((time_b == e_time) & (status_b == 1))
            n_at_risk[j] = np.sum((entry_b <= e_time) & (time_b >= e_time))

        # Compute the survival curve
        with np.errstate(divide="ignore", invalid="ignore"):
            survival_boot[i] = _km_fit(n_events=n_events, n_at_risk=n_at_risk)

        # Special case: if sufficiently late times didn't make it into our
        # bootstrap sample, then the risk set at those time is empty and the
        # resulting survival function estimates are nan (not a number). Instead,
        # make the survival probability at these times zero.
        survival_boot[i, n_at_risk == 0] = 0.

    variance = survival_boot.var(axis=0, ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma2 = variance / (survival ** 2)

    return variance, sigma2


def _km_ci(survival, variance, sigma2, conf_type, conf_level):
    """Compute Kaplan-Meier estimator confidence intervals."""
    # Standard normal quantiles for normal approximation confidence intervals
    quantile = st.norm.ppf((1 - conf_level) / 2)

    # Compute confidence intervals at the observed event times
    if conf_type == "linear":
        lower, upper = _km_ci_linear(survival=survival, variance=variance,
                                     quantile=quantile)
    elif conf_type == "log":
        lower, upper = _km_ci_log(survival=survival, sigma2=sigma2,
                                  quantile=quantile)
    elif conf_type == "log-log":
        lower, upper = _km_ci_log_log(survival=survival, sigma2=sigma2,
                                      quantile=quantile)
    elif conf_type == "logit":
        lower, upper = _km_ci_logit(survival=survival, sigma2=sigma2,
                                    quantile=quantile)
    elif conf_type == "arcsin":
        lower, upper = _km_ci_arcsin(survival=survival, sigma2=sigma2,
                                     quantile=quantile)
    else:
        # This should not be reachable
        raise RuntimeError(f"Invalid confidence interval type: {conf_type}.")

    # Force confidence interval bounds to be between 0 and 1
    with np.errstate(invalid="ignore"):
        lower = np.maximum(lower, 0.)
        upper = np.minimum(upper, 1.)

    return lower, upper


def _km_ci_linear(survival, variance, quantile):
    """Plain normal approximation CI."""
    error = quantile * np.sqrt(variance)
    lower = survival + error
    upper = survival - error
    return lower, upper


def _km_ci_log(survival, sigma2, quantile):
    """CI based on a delta method CI for log(S(t))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        error = quantile * np.sqrt(sigma2)
        lower = survival * np.exp(error)
        upper = survival * np.exp(-error)
    return lower, upper


def _km_ci_log_log(survival, sigma2, quantile):
    """CI based on a delta method CI for -log(-log(S(t)))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        error = quantile * np.sqrt(sigma2) / np.log(survival)
        lower = survival ** np.exp(error)
        upper = survival ** np.exp(-error)
    return lower, upper


def _km_ci_logit(survival, sigma2, quantile):
    """CI based on a delta method CI for log(S(t)/(1-S(t)))."""
    with np.errstate(invalid="ignore"):
        odds = survival / (1 - survival)
        error = np.exp(quantile * np.sqrt(sigma2) / (1 - survival))
        lower = 1 - 1 / (1 + odds * error)
        upper = 1 - 1 / (1 + odds / error)
    return lower, upper


def _km_ci_arcsin(survival, sigma2, quantile):
    """CI based on a delta method CI for arcsin(sqrt(S(t))."""
    with np.errstate(invalid="ignore"):
        arcsin = np.arcsin(np.sqrt(survival))
        odds = survival / (1 - survival)
        error = 0.5 * quantile * np.sqrt(odds * sigma2)
        lower = np.sin(np.maximum(0., arcsin + error)) ** 2
        upper = np.sin(np.minimum(np.pi / 2, arcsin - error)) ** 2
    return lower, upper
