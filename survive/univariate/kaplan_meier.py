"""The Kaplan-Meier estimator for non-parametric survival function estimation.

The Kaplan-Meier estimator is also called the product-limit estimator. For a
quick introduction, see Section 4.2 in Cox & Oakes (1984).

References
----------
    *  E. L. Kaplan and P. Meier. "Nonparametric estimation from incomplete
       observations". Journal of the American Statistical Association, Volume
       53, Issue 282 (1958), pp. 457--481. doi: https://doi.org/10.2307/2281868
    *  D. R. Cox and D. Oakes. Analysis of Survival Data. Chapman & Hall, London
       (1984), pp. ix+201.
"""

import numpy as np
import pandas as pd
import scipy.stats as st

from ..base import Model, Summary, Fittable, Predictor
from ..data import SurvivalData
from ..utils import check_data_1d, check_float


class KaplanMeier(Model, Fittable, Predictor):
    """Non-parametric survival function estimator for right-censored data.

    Properties
    ----------
    data : SurvivalData
        Survival data used to fit the estimator.
    summary : KaplanMeierSummary
        A summary of this Kaplan-Meier estimator.
    conf_type : str
        Type of confidence intervals for the survival function S(t) to report.
        Possible values:
            * "normal"
                Use a normal approximation to construct confidence intervals for
                S(t) directly.
            * "log-log"
                Use a normal approximation to construct confidence intervals for
                log(-log(S(t)) and transform this back to get confidence
                intervals for S(t).
    conf_level : float
        Confidence level of the confidence intervals.
    """
    model_type = "Kaplan-Meier estimator"
    data: SurvivalData

    # Internal versions of __init__() parameters
    _conf_type: str
    _conf_level: float

    # Distinct true failure times
    _fail: np.ndarray

    # Distinct censored times
    _censor: np.ndarray

    # Number of failures (deaths) at each true failure time
    _d: np.ndarray

    # Size of the risk set at each true failure time
    _r: np.ndarray

    # Estimate of the survival function at each observed failure time.
    _survival: np.ndarray

    @property
    def conf_type(self):
        """Type of confidence intervals for the survival function to report."""
        return self._conf_type

    @conf_type.setter
    def conf_type(self, conf_type):
        """Set the type of confidence interval."""
        if conf_type in ("normal", "log-log"):
            self._conf_type = conf_type
        else:
            raise ValueError(f"Invalid value for 'conf_type': {conf_type}")

    @property
    def conf_level(self):
        """Confidence level of the confidence intervals."""
        return self._conf_level

    @conf_level.setter
    def conf_level(self, conf_level):
        """Set the confidence level."""
        self._conf_level = check_float(conf_level, minimum=0., maximum=1.)

    def __init__(self, conf_type="log-log", conf_level=0.95):
        """Initialize the Kaplan-Meier estimator.

        Parameters
        ----------
        conf_type : str
            Type of confidence intervals for the survival function S(t) to
            report.
            Possible values:
                * "normal"
                    Use a normal approximation to construct confidence intervals
                    for S(t) directly.
                * "log-log"
                    Use a normal approximation to construct confidence intervals
                    for log(-log(S(t)) and transform this back to get confidence
                    intervals for S(t).
        conf_level : float
            Confidence level of the confidence intervals.
        """
        # Parameter validation is done in each parameter's setter method
        self.conf_type = conf_type
        self.conf_level = conf_level

    def fit(self, time, event=None, entry=None):
        """Fit the Kaplan-Meier estimator to survival data.

        Parameters
        ----------
        time : Lifetime or array-like of shape (n,)
            Observed times.
        event : array-like, of shape (n,), optional (default: None)
            Vector of 0's and 1's, 0 indicating a right-censored event, 1
            indicating a failure. This is ignored if `time` is already a
            Lifetime object.
        entry : array-like, one-dimensional, optional (default: None)
            Entry times of the observations (for left-truncated data). If not
            provided, the entry time for each observation is set to 0. This is
            ignored if `time` is already a Lifetime object.

        Returns
        -------
        self : KaplanMeier
            This KaplanMeier instance.
        """
        if isinstance(time, SurvivalData):
            self.data = time
        else:
            self.data = SurvivalData(time=time, event=event, entry=entry)

        # Extract values of interest from the survival data
        mask = (self.data.n_fail > 0)
        self._fail = self.data.time[mask]
        self._d = self.data.n_fail[mask]
        self._r = self.data.n_at_risk[mask]
        self._censor = self.data.time[self.data.n_censor > 0]

        # Compute the Kaplan-Meier product-limit estimator at the distinct
        # failure times
        self._survival = np.cumprod(1. - self._d / self._r)

        self.fitted = True
        return self

    def predict(self, time):
        """Estimate the survival probability at the given times.

        Parameters
        ----------
        time : array-like
            One-dimensional array of non-negative times.

        Returns
        -------
        prob : one-dimensional numpy.ndarray
            Estimated probabilities of exceeding the times in `time`.
        """
        self.check_fitted()
        time = check_data_1d(time)
        ind = np.searchsorted(self._fail, time, side="right")
        return np.concatenate(([1.], self._survival))[ind]

    def var(self, time):
        """Estimate the variance of the estimated survival probability at the
        given times using Greenwood’s formula (Greenwood 1926).

        Parameters
        ----------
        time : array-like
            One-dimensional array of non-negative times.

        Returns
        -------
        var : float or numpy.ndarray
            The variance estimate.

        References
        ----------
        M. Greenwood. "The natural duration of cancer". Reports on Public Health
            and Medical Subjects. Volume 33 (1926), pp. 1--26
        """
        self.check_fitted()
        time = check_data_1d(time)
        ind = np.searchsorted(self._fail, time, side="right")
        with np.errstate(divide="ignore", invalid="ignore"):
            # TODO: better variable names here
            a = self._d / self._r / (self._r - self._d)
            b = np.concatenate(([0.], np.cumsum(a)))
            return b[ind] * (self.predict(time) ** 2)

    def se(self, time):
        """Estimate the standard error of the estimated survival probability at
        the given times using Greenwood’s formula (Greenwood 1926).

        Parameters
        ----------
        time : array-like
            One-dimensional array of non-negative times.

        Returns
        -------
        std : float or numpy.ndarray
            The standard deviation estimate.

        References
        ----------
        M. Greenwood. "The natural duration of cancer". Reports on Public Health
            and Medical Subjects. Volume 33 (1926), pp. 1--26
        """
        return np.sqrt(self.var(time))

    def ci(self, time):
        """Compute confidence intervals for the survival probabilities.

        If conf_type is "normal", then the normal confidence interval
            max(S(t) + q * SE(S(t)), 0), min(S(t) - q * SE(S(t)), 1)
        is computed, where S(t) is the estimate of the survival function at time
        t, q is the (1-conf_level)/2 quantile of the standard normal
        distribution, and SE(S(t)) is the standard error of S(t) computed using
        Greenwood's formula (Greenwood 1926).

        If conf_type is "log-log", then the confidence interval is
            S(t) ** exp(c), S(t) ** exp(-c),
        where S(t) is the estimate of the survival function at time t, and
            c = q * SE(log(-log(S(t))),
        where q is the (1-conf_level)/2 quantile of the standard normal
        distribution and SE(log(-log(S(t))) is computed using the delta method.
        See Kalbfleisch & Prentice (2002).

        Parameters
        ----------
        time : array-like, one-dimensional
            One-dimensional array of non-negative times.

        Returns
        -------
        lower : one-dimensional numpy.ndarray
        upper : one-dimensional numpy.ndarray
            Lower and upper confidence interval bounds.

        References
        ----------
        M. Greenwood. "The natural duration of cancer". Reports on Public Health
            and Medical Subjects. Volume 33 (1926), pp. 1--26.
        John D. Kalbfleisch and Ross L. Prentice. The Statistical Analysis of
            Failure Time Data. Second Edition. Wiley (2002) pp. xiv+439.
        """
        self.check_fitted()

        time = check_data_1d(time)

        # Standard normal quantiles
        q = st.norm.ppf((1 - self.conf_level) / 2)

        # Estimated survival probabilities
        survival = self.predict(time)

        if self.conf_type == "log-log":
            # Log-log CI
            ind = np.searchsorted(self._fail, time, side="right")
            with np.errstate(divide="ignore", invalid="ignore"):
                # TODO: better variable names here
                a = self._d / self._r / (self._r - self._d)
                b = np.concatenate(([0.], np.cumsum(a)))
                c = q * np.sqrt(b[ind]) / np.log(survival)
                lower = survival ** (np.exp(c))
                upper = survival ** (np.exp(-c))
        else:
            # Normal approximation CI
            se = self.se(time)
            lower = survival + q * se
            upper = survival - q * se

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.maximum(lower, 0.), np.minimum(upper, 1.)

    def plot(self, ax=None, marker=True, marker_kwargs=None, ci=False,
             ci_kwargs=None, **kwargs):
        """Plot the Kaplan-Meier survival curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional (default: None)
            The axes on which to draw the line. If this is not specified, the
            current axis will be used.
        marker : bool, optional (default: True)
            If True, indicate the censored times by markers on the plot.
        marker_kwargs : dict, optional (default: None)
            Additional keyword parameters to pass to scatter() when
        ci : bool, optional (default: False)
            If True, draw point-wise confidence intervals (confidence bands).
        ci_kwargs : dict, optional (default: None)
            Additional keyword parameters to pass to step() or fill_between()
            when plotting the confidence band.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to the step() function.

        Returns
        -------
        The matplotlib.axes.Axes on which the curve was drawn.
        """
        self.check_fitted()

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        # Plot the survival curve
        x = self.data.time
        y = self.predict(x)
        params = dict(where="post", label=self.model_type, zorder=3)
        params.update(kwargs)
        p = ax.step(x, y, **params)

        # Mark the censored times
        if marker and self._censor.shape[0] != 0:
            color = p[0].get_color()
            marker_params = dict(marker="+", color=color, zorder=3)
            if marker_kwargs is not None:
                marker_params.update(marker_kwargs)
            xx = self._censor
            yy = self.predict(xx)
            ax.scatter(xx, yy, **marker_params)

        # Plot the confidence bands
        if ci:
            lower, upper = self.ci(x)
            label = f"{self.conf_level:.0%} {self.conf_type} C.I."
            color = p[0].get_color()
            alpha = 0.4 * params.get("alpha", 1.)
            ci_params = dict(color=color, alpha=alpha, label=label, step="post",
                             zorder=2)
            if ci_kwargs is not None:
                ci_params.update(ci_kwargs)
            ind = (~np.isnan(lower)) * (~np.isnan(upper))
            ax.fill_between(x[ind], lower[ind], upper[ind], **ci_params)

        # Configure axes
        ax.set(xlabel="Time", ylabel="Survival Probability")
        ax.autoscale(enable=True, axis="x")
        x_min, _ = ax.get_xlim()
        y_min, _ = ax.get_ylim()
        ax.set(xlim=(max(x_min, 0), None), ylim=(min(y_min, 0), None))

        return ax

    @property
    def summary(self):
        """Get a summary of this survival function estimator.

        Returns
        -------
        summary : SurvivalSummary
            The summary of this survival function estimator.
        """
        self.check_fitted()
        return KaplanMeierSummary(self)


class KaplanMeierSummary(Summary):
    """Summaries for Kaplan Meier survival function function estimators.

    Properties
    ----------
    model : KaplanMeier
        The Kaplan-Meier estimator being summarized.
    survival_table : pandas.DataFrame
        DataFrame summarizing the survival estimates at each observed time.
    """
    model: KaplanMeier

    @property
    def survival_table(self):
        """DataFrame summarizing the survival estimates at each observed time.

        Returns
        -------
        survival_table : pandas.DataFrame
            DataFrame summarizing the survival estimates at each observed time.
            Columns:
                * Time
                    An observed time.
                * At Risk
                    The number of individuals at risk at the observed time
                    (i.e., not failed or censored yet immediately before the
                    time).
                * Fail
                    The number of failures at the observed time.
                * Censor
                    The number of censored events at the observed time.
                * Survival
                    The estimated survival probability.
                * Std. Err.
                    The standard error of the survival probability estimate.
                * C.I. L
                    Lower confidence interval bound. The actual name of this
                    column will contain the confidence level.
                * C.I. R
                    Upper confidence interval bound. The actual name of this
                    column will contain the confidence level.
        """
        columns = ("Survival", "Std. Err.",
                   f"{self.model.conf_level:.0%} C.I. L",
                   f"{self.model.conf_level:.0%} C.I. R")
        survivor = self.model.predict(self.model.data.time)
        se = self.model.se(self.model.data.time)
        lower, upper = self.model.ci(self.model.data.time)
        estimate_summary = pd.DataFrame(dict(zip(columns,
                                                 (survivor, se, lower, upper))))
        return pd.concat((self.model.data.table, estimate_summary), axis=1)

    def __str__(self):
        """Return a string summary of the survivor function estimator."""
        header = super(KaplanMeierSummary, self).__str__()
        table = self.survival_table.to_string(index=False)
        n = self.model.data.n_at_risk[0]
        f = np.sum(self.model.data.n_fail)
        c = np.sum(self.model.data.n_censor)
        obs = f"{n} observations ({f} failures, {c} censored)"
        ci_info = f"{self.model.conf_level:.0%} confidence intervals are of " \
                  f"type '{self.model.conf_type}'."
        return f"{header}\n\n{obs}\n\n{table}\n\n{ci_info}"
