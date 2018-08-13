"""Breslow nonparametric survival function estimator."""

import numpy as np

from .base import NonparametricSurvival
from .nelson_aalen import NelsonAalen


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

    _conf_types = NelsonAalen._conf_types
    _var_types = NelsonAalen._var_types
    _tie_breaks = NelsonAalen._tie_breaks

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

        self.estimate_ = []
        self.estimate_var_ = []
        self.estimate_ci_lower_ = []
        self.estimate_ci_upper_ = []

        for i, group in enumerate(self._data.group_labels):
            # Extract Nelson-Aalen estimates for the current group
            na_estimate = nelson_aalen.estimate_[i]
            na_variance = nelson_aalen.estimate_var_[i]
            na_ci_lower = nelson_aalen.estimate_ci_lower_[i]
            na_ci_upper = nelson_aalen.estimate_ci_upper_[i]

            # The Breslow estimator is the exponential of the negative of the
            # Nelson-Aalen estimator
            breslow = np.exp(-na_estimate)
            self.estimate_.append(breslow)

            # Estimate the Breslow estimator variance using the delta method
            self.estimate_var_.append((breslow ** 2) * na_variance)

            # Get Breslow estimator confidence intervals by transforming the
            # Nelson-Aalen estimator confidence intervals
            self.estimate_ci_lower_.append(np.exp(-na_ci_lower))
            self.estimate_ci_upper_.append(np.exp(-na_ci_upper))

        self.fitted = True
        return self
