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

import itertools

import numpy as np
import pandas as pd
import scipy.stats as st

from ..base import Model, Summary, Fittable, Predictor
from ..base import SurvivalData
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
        Type of confidence intervals for the survival function estimate S(t) to
        report. Possible values:
            * "plain"
                Use a normal approximation to construct confidence intervals for
                S(t) directly.
            * "log"
                Derive confidence intervals for S(t) from normal approximation
                confidence intervals for the cumulative hazard function estimate
                -log(S(t)).
            * "log-log"
                Derive confidence intervals for S(t) from normal approximation
                confidence intervals for the log cumulative hazard function
                estimate log(-log(S(t))).
    conf_level : float
        Confidence level of the confidence intervals.
    """
    model_type = "Kaplan-Meier estimator"

    # Internal storage of the survival data
    _data: SurvivalData

    # Internal versions of __init__() parameters
    _conf_type: str
    _conf_level: float

    # Estimate of the survival function at each observed failure time within
    # each group.
    _survival: np.ndarray

    @property
    def conf_type(self):
        """Type of confidence intervals for the survival function to report."""
        return self._conf_type

    @conf_type.setter
    def conf_type(self, conf_type):
        """Set the type of confidence interval."""
        if conf_type in ("plain", "log", "log-log"):
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

    @property
    def data(self):
        """Survival data used to fit the estimator."""
        self.check_fitted()
        return self._data

    def __init__(self, conf_type="log-log", conf_level=0.95):
        """Initialize the Kaplan-Meier estimator.

        Parameters
        ----------
        conf_type : str
            Type of confidence intervals for the survival function estimate S(t)
            to report. Possible values:
                * "plain"
                    Use a normal approximation to construct confidence intervals
                    for S(t) directly.
                * "log"
                    Derive confidence intervals for S(t) from normal
                    approximation confidence intervals for the cumulative hazard
                    function estimate -log(S(t)).
                * "log-log"
                    Derive confidence intervals for S(t) from normal
                    approximation confidence intervals for the log cumulative
                    hazard function estimate log(-log(S(t))).
        conf_level : float
            Confidence level of the confidence intervals.
        """
        # Parameter validation is done in each parameter's setter method
        self.conf_type = conf_type
        self.conf_level = conf_level

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

        # Compute the Kaplan-Meier product-limit estimator at the distinct
        # failure times within each group
        self._survival = np.empty(self._data.n_groups, dtype=object)
        for i in range(self._data.n_groups):
            e = self._data.n_events[i]
            r = self._data.n_at_risk[i]
            self._survival[i] = np.cumprod(1. - e / r)

        self.fitted = True
        return self

    def predict(self, time, group=None):
        """Estimate the survival probability at the given times.

        Parameters
        ----------
        time : array-like
            One-dimensional array of non-negative times.
        group : group label or None, optional (default: None)
            Specify the group whose survival probability estimates should be
            returned. Ignored if there is only one group. If not specified,
            survival estimates for all the groups are returned.

        Returns
        -------
        prob : float or one-dimensional numpy.ndarray or pandas.DataFrame
            Estimated survival probabilities.
            Possible shapes:
                * If there is only one group or if a group is specified, then
                  this is either a float or a one-dimensional array depending on
                  whether the parameter `time` is a scalar or a one-dimensional
                  array with at least two elements, respectively.
                * If there is more than one group and no group is specified,
                  then this is a pandas.DataFrame with as many rows as entries
                  in `time` and one column for each group.
        """
        if group in self._data.groups:
            self.check_fitted()
            time = check_data_1d(time)
            i = np.flatnonzero(self._data.groups == group)[0]
            ind = np.searchsorted(self._data.time[i], time, side="right")
            prob = np.concatenate(([1.], self._survival[i]))[ind]
            return prob.item() if prob.size == 1 else prob
        elif self._data.n_groups == 1:
            return self.predict(time, group=self._data.groups[0])
        elif group is None:
            return pd.DataFrame({group: self.predict(time, group=group)
                                 for group in self._data.groups})
        else:
            raise ValueError(f"Not a known group label: {group}.")

    def _greenwood_sum(self, time, group_index):
        """Get the sum occurring in Greenwood's formula (Greenwood 1926) for the
        standard error of the survival function estimate and related estimates.

        Warning: this function does no validation of any kind.

        Parameters
        ----------
        time : array-like
            One-dimensional array of non-negative times.
        group_index : int
            The index of the group being looked at.

        Returns
        -------
        greenwood_sum : float or numpy.ndarray
            The sum used in Greenwood's formula.

        References
        ----------
        M. Greenwood. "The natural duration of cancer". Reports on Public Health
            and Medical Subjects. Volume 33 (1926), pp. 1--26
        """
        ind = np.searchsorted(self._data.time[group_index], time, side="right")
        e = self._data.n_events[group_index]
        r = self._data.n_at_risk[group_index]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.concatenate(([0], e / r / (r - e)))
        greenwood = np.cumsum(a)[ind]
        return greenwood.item() if greenwood.size == 1 else greenwood

    def var(self, time, group=None):
        """Estimate the variance of the estimated survival probability at the
        given times using Greenwood’s formula (Greenwood 1926).

        Parameters
        ----------
        time : array-like
            One-dimensional array of non-negative times.
        group : group label or None, optional (default: None)
            Specify the group whose variance estimates should be returned.
            Ignored if there is only one group. If not specified, variance
            estimates for all the groups are returned.

        Returns
        -------
        var : float or numpy.ndarray or pandas.DataFrame
            The variance estimate.
            Possible shapes:
                * If there is only one group or if a group is specified, then
                  this is either a float or a one-dimensional array depending on
                  whether the parameter `time` is a scalar or a one-dimensional
                  array with at least two elements, respectively.
                * If there is more than one group and no group is specified,
                  then this is a pandas.DataFrame with as many rows as entries
                  in `time` and one column for each group.

        References
        ----------
        M. Greenwood. "The natural duration of cancer". Reports on Public Health
            and Medical Subjects. Volume 33 (1926), pp. 1--26
        """
        if group in self._data.groups:
            self.check_fitted()
            time = check_data_1d(time)
            i = np.flatnonzero(self._data.groups == group)[0]
            prob = self.predict(time, group=group)
            with np.errstate(invalid="ignore"):
                return self._greenwood_sum(time, i) * (prob ** 2)
        elif self._data.n_groups == 1:
            return self.var(time, group=self._data.groups[0])
        elif group is None:
            return pd.DataFrame({group: self.var(time, group=group)
                                 for group in self._data.groups})
        else:
            raise ValueError(f"Not a known group label: {group}.")

    def se(self, time, group=None):
        """Estimate the standard error of the estimated survival probability at
        the given times using Greenwood’s formula (Greenwood 1926).

        Parameters
        ----------
        time : array-like
            One-dimensional array of non-negative times.
        group : group label or None, optional (default: None)
            Specify the group whose standard error estimates should be returned.
            Ignored if there is only one group. If not specified, standard error
            estimates for all the groups are returned.

        Returns
        -------
        std : float or numpy.ndarray or pandas.DataFrame
            The standard error estimate.
            Possible shapes:
                * If there is only one group or if a group is specified, then
                  this is either a float or a one-dimensional array depending on
                  whether the parameter `time` is a scalar or a one-dimensional
                  array with at least two elements, respectively.
                * If there is more than one group and no group is specified,
                  then this is a pandas.DataFrame with as many rows as entries
                  in `time` and one column for each group.

        References
        ----------
        M. Greenwood. "The natural duration of cancer". Reports on Public Health
            and Medical Subjects. Volume 33 (1926), pp. 1--26
        """
        return np.sqrt(self.var(time, group=group))

    def ci(self, time, group=None):
        """Compute confidence intervals for the survival function estimate S(t).

        In the following formulas, z denotes the (1-conf_level)/2 quantile of
        the standard normal distribution.

        If conf_type is "plain", then the confidence interval is
            [max(S(t) + z*SE, 0), min(S(t) - z*SE, 1)],
        where SE is the standard error estimate of S(t) computed using
        Greenwood's formula (Greenwood 1926).

        If conf_type is "log", then the confidence interval is
            [S(t) * exp(z*SE), min(S(t) * exp(-z*SE), 1)],
        where SE is the standard error estimate of the cumulative hazard
        function estimate -log(S(t)), computed using the delta method (similar
        to Greenwood's formula).

        If conf_type is "log-log", then the confidence interval is
           [S(t) ** exp(z*SE), S(t) ** exp(-z*SE)],
        where SE is the standard error estimate of the log cumulative hazard
        function estimate log(-log(S(t))), computed using the delta method
        (similar to Greenwood's formula).

        Parameters
        ----------
        time : array-like, one-dimensional
            One-dimensional array of non-negative times.
        group : group label or None, optional (default: None)
            Specify the group whose confidence intervals should be returned.
            Ignored if there is only one group. If not specified, confidence
            intervals for all the groups are returned.

        Returns
        -------
        lower : float or one-dimensional numpy.ndarray or pandas.DataFrame
        upper : float or one-dimensional numpy.ndarray or pandas.DataFrame
            Lower and upper confidence interval bounds.
            Possible shapes:
                * If there is only one group or if a group is specified, then
                  these are either floats or one-dimensional arrays depending
                  on whether the parameter `time` is a scalar or a
                  one-dimensional array with at least two elements,
                  respectively.
                * If there is more than one group and no group is specified,
                  then these are pandas.DataFrames with as many rows as entries
                  in `time` and one column for each group.

        References
        ----------
        M. Greenwood. "The natural duration of cancer". Reports on Public Health
            and Medical Subjects. Volume 33 (1926), pp. 1--26.
        """
        if group in self._data.groups:
            self.check_fitted()
            time = check_data_1d(time)

            # Get group index
            i = np.flatnonzero(self._data.groups == group)[0]

            # Standard normal quantile
            z = st.norm.ppf((1 - self.conf_level) / 2)

            # Estimated survival probabilities
            survival = self.predict(time, group=group)

            if self.conf_type == "plain":
                # Normal approximation CI
                se = self.se(time, group=group)
                lower = survival + z * se
                upper = survival - z * se
            elif self.conf_type == "log":
                # CI based on a CI for the cumulative hazard -log(S(t))
                with np.errstate(invalid="ignore"):
                    c = z * np.sqrt(self._greenwood_sum(time, i))
                    lower = survival * np.exp(c)
                    upper = survival * np.exp(-c)
            elif self.conf_type == "log-log":
                # CI based on a CI for log(-log(S(t)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_s = np.log(survival)
                    c = z * np.sqrt(self._greenwood_sum(time, i)) / log_s
                    lower = survival ** (np.exp(c))
                    upper = survival ** (np.exp(-c))
            else:
                # This should not be reachable
                raise ValueError(
                    f"Invalid confidence interval type: {self.conf_type}.")

            with np.errstate(invalid="ignore"):
                return np.maximum(lower, 0.), np.minimum(upper, 1.)
        elif self._data.n_groups == 1:
            return self.ci(time, group=self._data.groups[0])
        elif group is None:
            ls = np.empty(self._data.n_groups, dtype=object)
            us = np.empty(self._data.n_groups, dtype=object)
            for i, g in enumerate(self._data.groups):
                ls[i], us[i] = self.ci(time, g)
            lower = pd.DataFrame({g: l for g, l in zip(self._data.groups, ls)})
            upper = pd.DataFrame({g: u for g, u in zip(self._data.groups, us)})
            return lower, upper
        else:
            raise ValueError(f"Not a known group label: {group}.")

    def plot(self, *groups, ci=True, ci_kwargs=None, mark_censor=True,
             mark_censor_kwargs=None, legend=True, legend_kwargs=None,
             color=None, ax=None, **kwargs):
        """Plot the Kaplan-Meier survival curve.

        Parameters
        ----------
        *groups: list of group labels
            Specify the groups whose estimated survival curves should be
            plotted. If none are given, the survival curves for all groups are
            plotted.
        ci : bool, optional (default: False)
            If True, draw point-wise confidence intervals (confidence bands).
        ci_kwargs : dict, optional (default: None)
            Additional keyword parameters to pass to step() or fill_between()
            when plotting the confidence band.
        mark_censor : bool, optional (default: True)
            If True, indicate the censored times by markers on the plot.
        mark_censor_kwargs : dict, optional (default: None)
            Additional keyword parameters to pass to scatter() when marking
            censored times.
        legend : bool, optional (default: True)
            Indicates whether to display a legend for the plot.
        legend_kwargs : dict, optional (default: None)
            Keyword parameters to pass to legend().
        color : str or sequence, optional (default: None)
            Colors for each group's survival curve. This can be a string if only
            one curve is to be plotted.
        ax : matplotlib.axes.Axes, optional (default: None)
            The axes on which to draw the line. If this is not specified, the
            current axis will be used.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to the step() function when
            plotting survival curves.

        Returns
        -------
        The matplotlib.axes.Axes on which the curve was drawn.
        """
        self.check_fitted()

        if not groups:
            groups = self._data.groups

        if color is not None:
            if (len(groups) > 1 and ((not (isinstance(color, list)
                                           or isinstance(color, tuple)))
                                     or len(color) != len(groups))):
                raise ValueError("When plotting several curves, parameter "
                                 "'color' must be a list or tuple containing "
                                 "as many colors as groups.")
            if len(groups) == 1 and not isinstance(color, str):
                raise ValueError("When plotting a single curve, parameter "
                                 "'color' must be a string.")
            color = iter(color) if len(groups) > 1 else itertools.repeat(color)

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        # Plot the survival curves
        for group in groups:
            # Get group index
            i = np.flatnonzero(self._data.groups == group)[0]

            x = np.unique(np.concatenate((self._data.time[i],
                                          self._data.censor[i])))
            y = self.predict(x, group=group)
            label = f"{self.model_type}"
            if len(groups) > 1:
                label += f" ({group})"
            params = dict(where="post", label=label, zorder=3)
            if color is not None:
                params["color"] = next(color)
            params.update(kwargs)
            p = ax.step(x, y, **params)

            # Mark the censored times
            if mark_censor and self._data.censor[i].shape[0] != 0:
                c = p[0].get_color()
                marker_params = dict(marker="+", color=c, zorder=3)
                if mark_censor_kwargs is not None:
                    marker_params.update(mark_censor_kwargs)
                xx = self._data.censor[i]
                yy = self.predict(xx, group=group)
                ax.scatter(xx, yy, **marker_params)

            # Plot the confidence bands
            if ci:
                lower, upper = self.ci(x, group=group)
                label = f"{self.conf_level:.0%} {self.conf_type} C.I."
                if len(groups) > 1:
                    label += f" ({group})"
                c = p[0].get_color()
                alpha = 0.4 * params.get("alpha", 1.)
                ci_params = dict(color=c, alpha=alpha, label=label,
                                 step="post", zorder=2)
                if ci_kwargs is not None:
                    ci_params.update(ci_kwargs)
                ind = (~np.isnan(lower)) * (~np.isnan(upper))
                ax.fill_between(x[ind], lower[ind], upper[ind], **ci_params)

        # Configure axes
        ax.set(xlabel="Time", ylabel="Survival Probability")
        ax.autoscale(enable=True, axis="x")
        x_min, _ = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set(xlim=(max(x_min, 0), None), ylim=(min(y_min, 0), max(y_max, 1)))

        # Display the legend
        if legend:
            legend_params = dict(loc="best", frameon=True, shadow=True)
            if legend_kwargs is not None:
                legend_params.update(legend_kwargs)
            ax.legend(**legend_params)

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
    """
    model: KaplanMeier

    def table(self, group=None):
        """DataFrame summarizing the survival estimates at each observed time
        within a group.

        Parameters
        ----------
        group : group label, optional (default: None)
            Specify the group whose survival table should be returned. Ignored
            if there is only one group. If not specified, a list of the survival
            tables for all the groups is returned (if there is only one group,
            then a single table is returned).

        Returns
        -------
        tables : pandas.DataFrame or list of pandas.DataFrames
            If a group is specified or there is only one group in the data, then
            this is a pandas.DataFrame with the following columns.
                * time
                    The distinct true event times for that group.
                * at risk
                    Number of individuals at risk (i.e., entered but not yet
                    censored or failed) immediately before each distinct event
                    time for that group.
                * events
                    Number of failures/true events at each distinct event time
                    for that group.
                * survival
                    The estimated survival probability at each event time.
                * std. err.
                    The standard error of the survival probability estimate at
                    each event time.
                * c.i. lower
                    Lower confidence interval bound for the survival probability
                    at each event time. The actual name of this column will
                    contain the confidence level.
                * c.i. upper
                    Upper confidence interval bound for the survival probability
                    at each event time. The actual name of this column will
                    contain the confidence level.
            If no group is specified and there is more than one group total,
            then a list of such tables is returned (one for each group).
        """
        if group in self.model.data.groups:
            i = np.flatnonzero(self.model.data.groups == group)[0]
            columns = ("survival", "std. err.",
                       f"{self.model.conf_level:.0%} c.i. lower",
                       f"{self.model.conf_level:.0%} c.i. upper")
            survivor = self.model.predict(self.model.data.time[i], group=group)
            se = self.model.se(self.model.data.time[i], group=group)
            lower, upper = self.model.ci(self.model.data.time[i], group=group)
            bonus = pd.DataFrame(dict(zip(columns,
                                          (survivor, se, lower, upper))))
            return pd.concat((self.model.data.table(group=group), bonus),
                             axis=1)
        elif self.model.data.n_groups == 1:
            return self.table(group=self.model.data.groups[0])
        elif group is None:
            return [self.table(group=g) for g in self.model.data.groups]
        else:
            raise ValueError(f"Not a known group label: {group}.")

    def __str__(self):
        """Return a string summary of the survivor function estimator."""
        summary = super(KaplanMeierSummary, self).__str__()
        for group in self.model.data.groups:
            counts = self.model.data.counts.loc[[group]]
            summary += f"\n\n{counts.to_string()}"
            table = self.table(group)
            summary += f"\n\n{table.to_string(index=False)}"
        return summary
