"""The Kaplan-Meier nonparametric survival function estimator."""

import numpy as np
import scipy.stats as st

from .base import NonparametricSurvival
from .. import SurvivalData
from ..utils.validation import check_int


class KaplanMeier(NonparametricSurvival):
    """Kaplan-Meier survival function estimator.

    The Kaplan-Meier estimator [1]_ is also called the
    product-limit estimator. Much of this implementation is inspired by the R
    package ``survival`` [2]_.

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
    Confidence intervals for a survival probability estimate
    :math:`p=\hat{S}(t)` are computed using normal approximation confidence
    intervals for a strictly increasing differentiable transformation
    :math:`y=f(p)` using the delta method:

        If :math:`\mathrm{SE}(p)` is the standard error of :math:`p`, then the
        standard error of :math:`f(p)` is :math:`\mathrm{SE}(p) f^\prime(p)`.

    Consequently, a normal approximation confidence interval for :math:`f(p)` is

    .. math::

        f(p) \pm z \mathrm{SE}(p) f^\prime(p)

    where :math:`z` is the (1-`conf_level`)/2-quantile of the standard normal
    distribution. A confidence interval for :math:`p` is then

    .. math::

        f^{-1}(f(p) \pm z \mathrm{SE}(p) f^\prime(p))

    These confidence intervals were proposed in [6]_. We give a table of the
    supported transformations below.

        =========== ===========================
        `conf_type` :math:`f(p)`
        =========== ===========================
        "linear"    :math:`p`
        "log"       :math:`\log(p)`
        "log-log"   :math:`-\log(-\log(p))`
        "logit"     :math:`\log(p/(1-p))`
        "arcsin"    :math:`\sin^{-1}(\sqrt{p})`
        =========== ===========================

    Our implementation also shrinks the intervals to be between 0 and 1 if
    necessary.

    There are several supported ways of computing the standard error
    :math:`\mathrm{SE}(p)` of an estimated survival probability
    :math:`p = \hat{S}(t)`, each corresponding to a different value of
    `var_type`.

    1.  "greenwood" uses the classical Greenwood's formula [7]_.

    2.  "aalen-johansen" uses the Poisson moment approximation to the binomial
        suggested in [8]_. This method requires choosing how to handle tied
        event times by specifying the parameter `tie_break`. Possible values are

            * "discrete"
                Tied event times are possible and are treated as simultaneous.

            * "continuous"
                True event times almost surely don't coincide, and any observed
                ties are due to grouping or rounding. Tied event times will be
                treated as if each one occurred in succession, each one
                immediately following the previous one.

        This choice changes the definition of the Nelson-Aalen estimator
        increment, which consequently changes the definition of the
        Aalen-Johansen variance estimate. See Sections 3.1.3 and 3.2.2 in [9]_.

        This method is less frequently used than Greenwood's formula, and the
        two methods are usually close to each other numerically. However, [10]_
        recommends using Greenwood's formula because it is less biased and has
        comparable or lower mean squared error.

    3.  "bootstrap" uses the bootstrap (repeatedly sampling with replacement
        from the data and estimating the survival curve each time) to estimate
        the survival function variance [11]_.

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
    .. [6] Ørnulf Borgan and Knut Liestøl. "A note on confidence intervals and
        bands for the survival function based on transformations." Scandinavian
        Journal of Statistics. Volume 17, Number 1 (1990), pp. 35--41.
        `JSTOR <http://www.jstor.org/stable/4616153>`__.
    .. [7] M. Greenwood. "The natural duration of cancer". Reports on Public
        Health and Medical Subjects. Volume 33 (1926), pp. 1--26.
    .. [8] Odd O. Aalen and Søren Johansen. "An empirical transition matrix for
        non-homogeneous Markov chains based on censored observations."
        Scandinavian Journal of Statistics. Volume 5, Number 3 (1978),
        pp. 141--150. `JSTOR <http://www.jstor.org/stable/4615704>`__.
    .. [9] Odd O. Aalen, Ørnulf Borgan, and Håkon K. Gjessing. Survival and
        Event History Analysis. A Process Point of View. Springer-Verlag, New
        York (2008) pp. xviii+540.
        `DOI <https://doi.org/10.1007/978-0-387-68560-1>`__.
    .. [10] John P. Klein. "Small sample moments of some estimators of the
        variance of the Kaplan-Meier and Nelson-Aalen estimators." Scandinavian
        Journal of Statistics. Volume 18, Number 4 (1991), pp. 333--40.
        `JSTOR <http://www.jstor.org/stable/4616215>`__.
    .. [11] Bradley Efron. "Censored data and the bootstrap." Journal of the
        American Statistical Association. Volume 76, Number 374 (1981),
        pp. 312--19. `DOI <https://doi.org/10.2307/2287832>`__.
    """
    model_type = "Kaplan-Meier estimator"

    _conf_types = ("arcsin", "linear", "log", "log-log", "logit")

    # Types of variance estimators
    _var_types = ("aalen-johansen", "bootstrap", "greenwood")
    _var_type: str

    # How to handle tied event times for the Aalen-Johansen variance estimator
    _tie_breaks = ("continuous", "discrete")
    _tie_break: str

    # Number of bootstrap samples to draw
    _n_boot: int

    @property
    def var_type(self):
        """Type of variance estimate for the survival function to compute."""
        return self._var_type

    @var_type.setter
    def var_type(self, var_type):
        """Set the type of variance estimate."""
        if var_type in self._var_types:
            self._var_type = var_type
        else:
            raise ValueError(f"Invalid value for 'var_type': {var_type}.")

    @property
    def tie_break(self):
        """How to handle tied event times for the Aalen-Johansen variance
        estimator.
        """
        return self._tie_break

    @tie_break.setter
    def tie_break(self, tie_break):
        """Set the tie-breaking scheme."""
        if tie_break in self._tie_breaks:
            self._tie_break = tie_break
        else:
            raise ValueError(f"Invalid value for 'tie_break': {tie_break}.")

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
        self.estimate_ = []
        self.estimate_var_ = []
        self.estimate_ci_lower_ = []
        self.estimate_ci_upper_ = []
        for i, group in enumerate(self._data.group_labels):
            # d = number of events at an event time, y = size of the risk set at
            # an event time
            d = self._data.events[group].n_events
            y = self._data.events[group].n_at_risk

            # Product-limit survival probability estimates
            self.estimate_.append(np.cumprod(1. - d / y))

            # In the following block, the variable ``sigma2`` is the variance
            # estimate divided by the square of the survival function
            # estimate. It arises again in our confidence interval computations
            # later.
            if self._var_type == "bootstrap":
                # Estimate the survival function variance using the bootstrap
                var = _km_var_boot(data=self._data, index=i,
                                   random_state=self._random_state,
                                   n_boot=self.n_boot)
                self.estimate_var_.append(var)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sigma2 = self.estimate_var_[i] / (self.estimate_[i] ** 2)
            else:
                # Estimate the survival function variance using Greenwood's
                # formula or the Aalen-Johansen method
                if self._var_type == "greenwood":
                    # Greenwood's formula
                    with np.errstate(divide="ignore"):
                        sigma2 = np.cumsum(d / y / (y - d))
                elif self._var_type == "aalen-johansen":
                    # Aalen-Johansen estimate
                    if self._tie_break == "discrete":
                        sigma2 = np.cumsum(d / (y ** 2))
                    elif self._tie_break == "continuous":
                        # Increments of sum in equation (3.14) on page 84 of
                        # Aalen, Borgan, & Gjessing (2008)
                        inc = np.empty(len(d), dtype=np.float_)
                        for j in range(len(d)):
                            inc[j] = np.sum(1 / (y[j] - np.arange(d[j])) ** 2)
                        sigma2 = np.cumsum(inc)
                    else:
                        # This should not be reachable
                        raise RuntimeError(
                            f"Invalid tie-breaking scheme: {self._tie_break}.")
                else:
                    # This should not be reachable
                    raise RuntimeError(
                        f"Invalid variance type: {self._var_type}.")

                with np.errstate(invalid="ignore"):
                    self.estimate_var_.append((self.estimate_[i] ** 2) * sigma2)

            # Standard normal quantile for confidence intervals
            z = st.norm.ppf((1 - self.conf_level) / 2)

            # Compute confidence intervals at the observed event times
            if self._conf_type == "linear":
                # Normal approximation CI
                c = z * np.sqrt(self.estimate_var_[i])
                lower = self.estimate_[i] + c
                upper = self.estimate_[i] - c
            elif self._conf_type == "log":
                # CI based on a delta method CI for log(S(t))
                with np.errstate(invalid="ignore"):
                    c = z * np.sqrt(sigma2)
                    lower = self.estimate_[i] * np.exp(c)
                    upper = self.estimate_[i] * np.exp(-c)
            elif self._conf_type == "log-log":
                # CI based on a delta method CI for -log(-log(S(t)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    c = z * np.sqrt(sigma2) / np.log(self.estimate_[i])
                    lower = self.estimate_[i] ** np.exp(c)
                    upper = self.estimate_[i] ** np.exp(-c)
            elif self._conf_type == "logit":
                # CI based on a delta method CI for log(S(t)/(1-S(t)))
                with np.errstate(invalid="ignore"):
                    odds = self.estimate_[i] / (1 - self.estimate_[i])
                    c = z * np.sqrt(sigma2) / (1 - self.estimate_[i])
                    lower = 1 - 1 / (1 + odds * np.exp(c))
                    upper = 1 - 1 / (1 + odds * np.exp(-c))
                pass
            elif self._conf_type == "arcsin":
                # CI based on a delta method CI for arcsin(sqrt(S(t))
                with np.errstate(invalid="ignore"):
                    arcsin = np.arcsin(np.sqrt(self.estimate_[i]))
                    odds = self.estimate_[i] / (1 - self.estimate_[i])
                    c = 0.5 * z * np.sqrt(odds * sigma2)
                    lower = np.sin(np.maximum(0., arcsin + c)) ** 2
                    upper = np.sin(np.minimum(np.pi / 2, arcsin - c)) ** 2
            else:
                # This should not be reachable
                raise RuntimeError(
                    f"Invalid confidence interval type: {self._conf_type}.")

            # Force confidence interval bounds to be between 0 and 1
            with np.errstate(invalid="ignore"):
                self.estimate_ci_lower_.append(np.maximum(lower, 0.))
                self.estimate_ci_upper_.append(np.minimum(upper, 1.))

            # Make sure that variance estimates and confidence intervals are NaN
            # when the estimated survival probability is zero
            ind_zero = (self.estimate_[i] == 0.)
            self.estimate_var_[i][ind_zero] = np.nan
            self.estimate_ci_lower_[i][ind_zero] = np.nan
            self.estimate_ci_upper_[i][ind_zero] = np.nan

        self.fitted = True
        return self


def _km_var_boot(data: SurvivalData, index, random_state, n_boot):
    """Estimate Kaplan-Meier survival function variance using the bootstrap.

    Parameters
    ----------
    data : SurvivalData
        Survival data used to fit the Kaplan-Meier estimator.

    index : int
        The group index.

    random_state : numpy.random.RandomState
        Random number generator.

    n_boot : int
        Number of bootstrap samples to draw.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of survival function variance estimates at each
        observed event time.
    """
    # Extract observed times, censoring indicators, and entry times for the
    # specified group
    ind = (data.group == data.group_labels[index])
    time = np.asarray(data.time[ind])
    status = np.asarray(data.status[ind])
    entry = np.asarray(data.entry[ind])

    # Distinct true event times
    events = np.unique(time[status == 1])

    # n = sample size, k = number of distinct true events
    n = len(time)
    k = len(events)

    # Initialize array of bootstrap Kaplan-Meier survival function estimates at
    # the observed true event times
    survival_boot = np.empty(shape=(n_boot, k), dtype=np.float_)

    # The bootstrap
    for i in range(n_boot):
        # Draw a bootstrap sample
        ind_boot = random_state.choice(n, size=n, replace=True)
        time_boot = time[ind_boot]
        status_boot = status[ind_boot]
        entry_boot = entry[ind_boot]

        # e = number of events at an event time, r = size of the risk set at an
        # event time
        e = np.empty(shape=(k,), dtype=np.int_)
        r = np.empty(shape=(k,), dtype=np.int_)
        for j, t in enumerate(events):
            e[j] = np.sum((time_boot == t) & (status_boot == 1))
            r[j] = np.sum((entry_boot <= t) & (time_boot >= t))

        # Compute the survival curve
        with np.errstate(divide="ignore", invalid="ignore"):
            survival_boot[i] = np.cumprod(1. - e / r)

        # Special case: if sufficiently late times didn't make it into our
        # bootstrap sample, then the risk set at those time is empty and the
        # resulting survival function estimates are nan (not a number). Instead,
        # make the survival probability at these times zero.
        survival_boot[i, r == 0] = 0.

    return survival_boot.var(axis=0, ddof=1)
