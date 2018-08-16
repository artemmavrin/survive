"""Abstract base class for univariate estimators."""

import abc

import numpy as np
import pandas as pd

from ..base import Model, Fittable, Predictor, Summary
from ..survival_data import SurvivalData
from ..utils import add_legend
from ..utils import check_bool, check_colors, check_data_1d, check_float


class NonparametricEstimator(Model, Fittable, Predictor):
    """Abstract base class for nonparametric estimators."""

    # The quantity being estimated (e.g., survival function, cumulative hazard,
    # hazard rate, etc.)
    _estimand: str

    # Estimate at time zero
    _estimate0: np.float_

    # Types of confidence intervals available
    _conf_types: tuple

    # Types of variance estimates available
    _var_types: tuple

    # How to handle tied event times
    _tie_breaks = ("continuous", "discrete")

    # Internal storage of the survival data
    _data: SurvivalData

    # Internal versions of __init__() parameters
    _conf_type: str
    _conf_level: float
    _var_type: str
    _tie_break: str

    # Estimate at each distinct event time for each group
    estimate_: dict

    # Estimate of the estimator's variance at each distinct event time for each
    # group
    var_: dict

    # Upper and lower confidence interval bounds for the estimate at each
    # distinct event time for each group
    ci_lower_: dict
    ci_upper_: dict

    @property
    def conf_type(self):
        """Type of confidence intervals to report.

        Returns
        -------
        conf_type : str
            The type of confidence interval.
        """
        return self._conf_type

    @conf_type.setter
    def conf_type(self, conf_type):
        """Set the type of confidence interval."""
        if conf_type in self._conf_types:
            self._conf_type = conf_type
        else:
            raise ValueError(f"Invalid value for 'conf_type': {conf_type}.")

    @property
    def conf_level(self):
        """Confidence level of the confidence intervals.

        Returns
        -------
        conf_level : float
            The confidence level.
        """
        return self._conf_level

    @conf_level.setter
    def conf_level(self, conf_level):
        """Set the confidence level."""
        self._conf_level = check_float(conf_level, minimum=0., maximum=1.)

    @property
    def var_type(self):
        """Type of variance estimate to compute."""
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
        """How to handle tied event times.
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
    def data_(self) -> SurvivalData:
        """Survival data used to fit the estimator.

        This :class:`property` is only available after fitting.

        Returns
        -------
        data : SurvivalData
            The :class:`survive.SurvivalData` instance used to fit the
            estimator.
        """
        self.check_fitted()
        return self._data

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def predict(self, time, *, return_se=False, return_ci=False):
        """Compute estimates.

        Parameters
        ----------
        time : array-like
            One-dimensional array of times at which to make estimates.

        return_se : bool, optional
            If True, also return standard error estimates.

        return_ci : bool, optional
            If True, also return confidence intervals.

        Returns
        -------
        estimate : pandas.DataFrame
            DataFrame of estimates. Each columns represents a group, and each
            row represents an entry of `time`.

        std_err : pandas.DataFrame, optional
            Standard errors of the estimates. Same shape as `estimate`. Returned
            only if `return_se` is True.

        lower : pandas.DataFrame, optional
            Lower confidence interval bounds. Same shape as `estimate`. Returned
            only if `return_ci` is True.

        upper : pandas.DataFrame, optional
            Upper confidence interval bounds. Same shape as `estimate`. Returned
            only if `return_ci` is True.
        """
        # Validation
        self.check_fitted()
        time = check_data_1d(time, keep_pandas=False)
        return_ci = check_bool(return_ci)
        return_se = check_bool(return_se)

        groups = self._data.group_labels

        # Initialize arrays
        estimate = np.empty((time.shape[0], self._data.n_groups),
                            dtype=self.estimate_[groups[0]].dtype)
        std_err = np.empty(estimate.shape, dtype=np.float_)
        lower = np.empty(estimate.shape,
                         dtype=self.ci_lower_[groups[0]].dtype)
        upper = np.empty(estimate.shape,
                         dtype=self.ci_upper_[groups[0]].dtype)

        # Compute everything within each group
        for j, group in enumerate(self._data.group_labels):
            # Figure out where to insert the times in ``time`` within the
            # observed event times for the group
            ind = np.searchsorted(self._data.events[group].time, time,
                                  side="right")

            # Compute everything requested for the current group
            estimate[:, j] = \
                np.concatenate(([self._estimate0], self.estimate_[group]))[ind]
            if return_se:
                std_err[:, j] = \
                    np.concatenate(([0.], np.sqrt(self.var_[group])))[ind]
            if return_ci:
                lower[:, j] = np.concatenate(([self._estimate0],
                                              self.ci_lower_[group]))[ind]
                upper[:, j] = np.concatenate(([self._estimate0],
                                              self.ci_upper_[group]))[ind]

        # Create the DataFrame of estimates
        columns = pd.Index(self._data.group_labels, name="group")
        index = pd.Index(time, name="time")
        estimate = pd.DataFrame(estimate, columns=columns, index=index)

        # Return the correct things
        if not (return_se or return_ci):
            return estimate

        # Create all other necessary DataFrames and return
        out = (estimate,)
        if return_se:
            out += (pd.DataFrame(std_err, columns=columns, index=index),)
        if return_ci:
            out += (pd.DataFrame(lower, columns=columns, index=index),
                    pd.DataFrame(upper, columns=columns, index=index))
        return out

    @property
    def summary(self):
        """Get a summary of this estimator.

        Returns
        -------
        summary : NonparametricEstimatorSummary
            The summary of this estimator.

        See Also
        --------
        survive.nonparametric.NonparametricEstimatorSummary
        """
        self.check_fitted()
        return NonparametricEstimatorSummary(self)

    def plot(self, *groups, ci=True, ci_style="fill", ci_kwargs=None,
             mark_censor=True, mark_censor_kwargs=None, legend=True,
             legend_kwargs=None, colors=None, palette=None, ax=None, **kwargs):
        """Plot the estimates.

        Parameters
        ----------
        *groups : list of group labels
            Specify the groups whose curves should be plotted. If none are
            given, the curves for all groups are plotted.

        ci : bool, optional
            If True, draw pointwise confidence intervals.

        ci_style : {"fill", "lines"}, optional
            Specify how to draw the confidence intervals. If `ci_style` is
            "fill", the region between the lower and upper confidence interval
            curves will be filled. If `ci_style` is "lines", only the lower and
            upper curves will be drawn (this is inspired by the style of
            confidence intervals drawn by `plot.survfit` in the R package
            `survival`).

        ci_kwargs : dict, optional
            Additional keyword parameters to pass to
            :meth:`~matplotlib.axes.Axes.fill_between` (if `ci_style` is "fill")
            or :meth:`~matplotlib.axes.Axes.step` (if `ci_style` is "lines")
            when plotting the pointwise confidence intervals.

        mark_censor : bool, optional
            If True, indicate the censored times by markers on the plot.

        mark_censor_kwargs : dict, optional
            Additional keyword parameters to pass to
            :meth:`~matplotlib.axes.Axes.scatter` when marking censored times.

        legend : bool, optional
            Indicates whether to display a legend for the plot.

        legend_kwargs : dict, optional
            Keyword parameters to pass to :meth:`~matplotlib.axes.Axes.legend`.

        colors : list or tuple or dict or str, optional
            Colors for each group. This is ignored if `palette` is provided.
            This can be a sequence of valid matplotlib colors to cycle through,
            or a dictionary mapping group labels to matplotlib colors, or the
            name of a matplotlib colormap.

        palette : str, optional
            Name of a seaborn color palette. Requires seaborn to be installed.
            Setting a color palette overrides the `colors` parameter.

        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If this is not specified, the current
            axes will be used.

        **kwargs : keyword arguments
            Additional keyword arguments to pass to
            :meth:`~matplotlib.axes.Axes.step` when plotting the estimates.

        Returns
        -------
        matplotlib.axes.Axes
            The :class:`~matplotlib.axes.Axes` on which the plot was drawn.
        """
        self.check_fitted()

        # Validate groups
        if not groups:
            # Plot curves for all groups
            groups = self._data.group_labels
        else:
            # Ensure that the group names are valid
            for group in groups:
                if group not in self._data.group_labels:
                    raise ValueError(f"Not a known group label: {group}.")

        # Get colors to cycle through
        colors = check_colors(colors, n_colors=len(groups), keys=groups,
                              palette=palette)

        # Get current axes if axes are not specified
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        # Times in the sample
        time = np.unique(self._data.time)

        ci = check_bool(ci)
        if ci:
            estimate, lower, upper = self.predict(time, return_ci=True)
        else:
            estimate = self.predict(time)
            lower = upper = None

        # Plot the estimate curves
        for group in groups:
            # Parameters for the plot
            color = next(colors)
            curve_label = f"{self.model_type}"
            if len(groups) > 1:
                curve_label += f" ({group})"
            curve_params = dict(where="post", label=curve_label, zorder=3)
            if color is not None:
                curve_params["color"] = color
            curve_params.update(kwargs)

            # Plot this group's curve
            p = ax.step(time, estimate[group], **curve_params)

            # Mark the censored times
            if mark_censor and self._data.censor[group].shape[0] != 0:
                c = p[0].get_color()
                marker_params = dict(marker="+", color=c, zorder=3, label=None)
                if mark_censor_kwargs is not None:
                    marker_params.update(mark_censor_kwargs)
                censor_times = self._data.censor[group].time
                censor_estimate = np.empty(censor_times.shape, dtype=np.float_)
                for j, t in enumerate(censor_times):
                    censor_estimate[j] = estimate[group].loc[time == t]
                ax.scatter(censor_times, censor_estimate, **marker_params)

            # Plot the confidence intervals
            if ci:
                ci_label = f"{self.conf_level:.0%} {self.conf_type} C.I."
                if len(groups) > 1:
                    ci_label += f" ({group})"
                c = p[0].get_color()

                if ci_style == "fill":
                    alpha = 0.4 * curve_params.get("alpha", 1.)
                    ci_params = dict(color=c, alpha=alpha, label=ci_label,
                                     step="post", zorder=2)
                    if ci_kwargs is not None:
                        ci_params.update(ci_kwargs)
                    ax.fill_between(time, lower[group], upper[group],
                                    **ci_params)
                elif ci_style == "lines":
                    ci_params = dict(color=c, label=ci_label, where="post",
                                     zorder=2, linestyle="--")
                    if ci_kwargs is not None:
                        ci_params.update(ci_kwargs)
                    ax.step(time, lower[group], **ci_params)
                    ci_params["label"] = None
                    ax.step(time, np.asarray(upper[group]), **ci_params)
                else:
                    raise ValueError(f"Unknown CI style: {ci_style}.")

        # Configure axes
        ax.set(xlabel="time", ylabel=self._estimand)
        x_min, _ = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set(xlim=(max(x_min, 0), None), ylim=(min(y_min, 0), max(y_max, 1)))

        # Display the legend
        if legend and len(groups) > 1:
            add_legend(ax, legend_kwargs)

        return ax


class NonparametricSurvival(NonparametricEstimator):
    """Abstract base class for nonparametric survival function estimators."""
    _estimand = "survival function"
    _estimate0 = np.float_(1.)

    # Tolerance for checking if a probability level "exactly" equals a survival
    # probability when computing quantiles of the time-to-event distribution.
    # This is to counteract round-off error encountered when computing the
    # survival function estimates.
    _quantile_tol = np.sqrt(np.finfo(np.float_).eps)

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def quantile(self, prob, *, return_ci=False):
        r"""Empirical quantile estimates for the time-to-event distribution.

        Parameters
        ----------
        prob : array-like
            One-dimensional array of values between 0 and 1 representing the
            probability levels of the desired quantiles.

        return_ci : bool, optional
            Specify whether to return confidence intervals for the quantile
            estimates.

        Returns
        -------
        quantiles : pandas.DataFrame
            The quantile estimates. Rows are indexed by the entries of `time`
            and columns are indexed by the model's group labels. Entries for
            probability levels for which the quantile estimate is not defined
            are nan (not a number).

        lower : pandas.DataFrame, optional
            Lower confidence interval bounds for the quantile estimates.
            Returned only if `return_ci` is True. Same shape as `quantiles`.

        upper : pandas.DataFrame, optional
            Upper confidence interval bounds for the quantile estimates.
            Returned only if `return_ci` is True. Same shape as `quantiles`.

        Notes
        -----
        For a probability level :math:`p` between 0 and 1, the empirical
        :math:`p`-quantile of the time-to-event distribution with estimated
        survival function :math:`\widehat{S}(t)` is defined to be the time at
        which the horizontal line at height :math:`1-p` intersects with the
        estimated survival curve. If such a time is not unique, then instead
        there is a time interval on which the estimated survival curve is flat
        and coincides with the horizontal line at height :math:`1-p`. In this
        case the midpoint of this interval is taken to be the empirical
        :math:`p`-quantile estimate (this is just one of many possible
        conventions, and the one used by the R package ``survival`` [1]_). If
        the survival function estimate never gets as low as :math:`1-p`, then
        the :math:`p`-quantile cannot be estimated.

        The confidence intervals computed here are based on finding the time at
        which the horizontal line at height :math:`1-p` intersects the upper
        and lower confidence interval for :math:`\widehat{S}(t)`. This mimics
        the implementation in the R package ``survival`` [1]_, which is based on
        the confidence interval construction in [2]_.

        References
        ----------
        .. [1] Terry M. Therneau. A Package for Survival Analysis in S. version
            2.38 (2015). `CRAN <https://CRAN.R-project.org/package=survival>`__.
        .. [2] Ron Brookmeyer and John Crowley. "A Confidence Interval for the
            Median Survival Time." Biometrics, Volume 38, Number 1 (1982),
            pp. 29--41. `DOI <https://doi.org/10.2307/2530286>`__.
        """
        self.check_fitted()
        return_ci = check_bool(return_ci)
        prob = check_data_1d(prob, keep_pandas=False)
        if not np.all((prob >= 0) * (prob <= 1)):
            raise ValueError("Probability levels must be between zero and one.")

        # Initialize all arrays
        quantiles = np.empty((prob.shape[0], self._data.n_groups),
                             dtype=np.float_)
        lower = np.empty(quantiles.shape, dtype=np.float_)
        upper = np.empty(quantiles.shape, dtype=np.float_)

        # Compute quantile estimates and confidence intervals if necessary
        for j, group in enumerate(self._data.group_labels):
            mask = (self._data.group == group)
            quantiles[:, j] = \
                _find_intersection(p=prob, y=self.estimate_[group],
                                   tol=self._quantile_tol,
                                   c_times=self._data.censor[group].time,
                                   e_times=self._data.events[group].time,
                                   entry=self._data.entry[mask])

            if return_ci:
                lower[:, j] = \
                    _find_intersection(p=prob, y=self.ci_lower_[group],
                                       tol=self._quantile_tol,
                                       c_times=self._data.censor[group].time,
                                       e_times=self._data.events[group].time,
                                       entry=self._data.entry[mask])
                upper[:, j] = \
                    _find_intersection(p=prob, y=self.ci_upper_[group],
                                       tol=self._quantile_tol,
                                       c_times=self._data.censor[group].time,
                                       e_times=self._data.events[group].time,
                                       entry=self._data.entry[mask])

        columns = pd.Index(self._data.group_labels, name="group")
        index = pd.Index(prob, name="prob")
        quantiles = pd.DataFrame(quantiles, columns=columns, index=index)

        if not return_ci:
            return quantiles

        lower = pd.DataFrame(lower, columns=columns, index=index)
        upper = pd.DataFrame(upper, columns=columns, index=index)
        return quantiles, lower, upper


def _find_intersection(p, y, tol, c_times, e_times, entry):
    """Find points of intersection of estimated curve `y` with probability level
    1-`p`.

    Used for quantile estimation.

    Parameters
    ----------
    p : numpy.ndarray
        Probability levels.
    y : numpy.ndarray
        Numbers between 0 and 1 defined at each event time.
    tol : float
        Tolerance for exact equality between `p` and `y`.
    c_times : pandas.Series
        Ordered distinct censored times.
    e_times : pandas.Series
        Ordered distinct event times.
    entry : numpy.ndarray
        List of entry times.
    """
    # Find intersection of horizontal line at height 1-p with y curve
    y = np.concatenate(([0.], 1 - y))
    ind1 = np.searchsorted(y - tol, p)
    ind2 = np.searchsorted(y + tol, p)

    # Find out whether the last time was censored or not
    if c_times.shape[0] > 0 and c_times.iloc[-1] > e_times.iloc[-1]:
        last = c_times.iloc[-1]
    else:
        last = e_times.iloc[-1]

    x = np.concatenate(([0.], e_times, [last]))
    x = 0.5 * (x[ind1] + x[ind2])

    # Special cases
    with np.errstate(invalid="ignore"):
        x[p < tol] = np.min(entry)
        x[p > np.nanmax(y) + tol] = np.nan

    return x


class NonparametricEstimatorSummary(Summary):
    """Summaries for nonparametric estimators."""

    model: NonparametricEstimator

    def table(self, group):
        """DataFrame summarizing the the nonparametric estimates for a group.

        Parameters
        ----------
        group : group label
            Specify the group whose summary table should be returned.

        Returns
        -------
        table : pandas.DataFrame
            Summary table with the following columns.

            time
                The distinct event times for the group.

            events
                Number of events at each distinct event time for the group.

            at risk
                Number of individuals at risk (i.e., entered but not yet
                censored or failed) immediately before each distinct event time
                for the group.

            estimate
                The estimate at each event time.

            std. err.
                The standard error of the estimate at each event time.

            c.i. lower
                Lower confidence interval bound for the estimate at each event
                time. The actual name of this column will contain the confidence
                level.

            c.i. upper
                Upper confidence interval bound for the estimate at each event
                time. The actual name of this column will contain the confidence
                level.
        """
        if group not in self.model.data_.group_labels:
            raise ValueError(f"Not a known group label: {group}.")

        columns = ("time", "events", "at risk", "estimate", "std. error",
                   f"{self.model.conf_level:.0%} c.i. lower",
                   f"{self.model.conf_level:.0%} c.i. upper")

        time = self.model.data_.events[group].time
        events = self.model.data_.events[group].n_events
        at_risk = self.model.data_.events[group].n_at_risk

        estimate = self.model.estimate_[group]
        std_err = np.sqrt(self.model.var_[group])
        ci_lower = self.model.ci_lower_[group]
        ci_upper = self.model.ci_upper_[group]

        table_dict = dict(zip(columns, (time, events, at_risk, estimate,
                                        std_err, ci_lower, ci_upper)))
        return pd.DataFrame(table_dict)

    def __repr__(self):
        """Return a string representation of the summary."""
        summary = super(NonparametricEstimatorSummary, self).__repr__()
        describe = self.model.data_.describe
        for group in self.model.data_.group_labels:
            if self.model.data_.n_groups > 1:
                summary += f"\n\n{group}"
            summary += f"\n\n{describe.loc[[group]].to_string(index=False)}"
            summary += f"\n\n{self.table(group).to_string(index=False)}"
        return summary
