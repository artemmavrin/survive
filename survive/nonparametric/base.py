"""Abstract base class for univariate estimators."""

import abc

import numpy as np
import pandas as pd

from .. import SurvivalData
from ..base import Model, Fittable, Predictor, Summary
from ..utils.validation import (check_bool, check_colors, check_data_1d,
                                check_float)


class NonparametricEstimator(Model, Fittable, Predictor):
    """Abstract base class for nonparametric estimators."""

    # The quantity being estimated (e.g., survival function, cumulative hazard,
    # hazard rate, etc.)
    _estimand: str

    # Types of confidence intervals available
    _conf_types: tuple

    # Internal storage of the survival data
    _data: SurvivalData

    # Internal versions of __init__() parameters
    _conf_type: str
    _conf_level: float

    # Estimate at each distinct event time for each group
    estimate_: list
    estimate_var_: list
    estimate_ci_lower_: list
    estimate_ci_upper_: list

    # Estimate at time zero
    _estimate0: np.float_

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
        """Return estimates of at the given times.

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

        # Initialize arrays
        estimate = np.empty((time.shape[0], self._data.n_groups),
                            dtype=self.estimate_[0].dtype)
        std_err = np.empty(estimate.shape, dtype=np.float_)
        lower = np.empty(estimate.shape, dtype=self.estimate_ci_lower_[0].dtype)
        upper = np.empty(estimate.shape, dtype=self.estimate_ci_upper_[0].dtype)

        # Compute everything within each group
        for j, group in enumerate(self._data.group_labels):
            # Figure out where to insert the times in ``time`` within the
            # observed event times for the group
            ind = np.searchsorted(self._data.events[group].time, time,
                                  side="right")

            # Compute everything requested for the current group
            estimate[:, j] = \
                np.concatenate(([self._estimate0], self.estimate_[j]))[ind]
            if return_se:
                std_err[:, j] = \
                    np.concatenate(([0.], np.sqrt(self.estimate_var_[j])))[ind]
            if return_ci:
                lower[:, j] = np.concatenate(([self._estimate0],
                                              self.estimate_ci_lower_[j]))[ind]
                upper[:, j] = np.concatenate(([self._estimate0],
                                              self.estimate_ci_upper_[j]))[ind]

        # Create the DataFrame of estimates
        columns = pd.Index(self._data.group_labels, name="group")
        index = pd.Index(time, name="time")
        estimate = pd.DataFrame(estimate, columns=columns, index=index)

        # Return the correct things
        if not (return_se or return_ci):
            return estimate
        else:
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

    def plot(self, *groups, ci=True, ci_kwargs=None, mark_censor=True,
             mark_censor_kwargs=None, legend=True, legend_kwargs=None,
             colors=None, palette=None, ax=None, **kwargs):
        """Plot the nonparametric estimates.

        Parameters
        ----------
        *groups : list of group labels
            Specify the groups whose curves should be plotted. If none are
            given, the curves for all groups are plotted.

        ci : bool, optional
            If True, draw point-wise confidence intervals (confidence bands).

        ci_kwargs : dict, optional
            Additional keyword parameters to pass to
            :func:`matplotlib.axes.Axes.fill_between` when plotting the
            confidence band.

        mark_censor : bool, optional
            If True, indicate the censored times by markers on the plot.

        mark_censor_kwargs : dict, optional
            Additional keyword parameters to pass to
            :func:`matplotlib.axes.Axes.scatter` when marking censored times.

        legend : bool, optional
            Indicates whether to display a legend for the plot.

        legend_kwargs : dict, optional
            Keyword parameters to pass to :func:`matplotlib.axes.Axes.legend`.

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
            :func:`matplotlib.axes.Axes.step` when plotting the estimates.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the plot was drawn.
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

        # Validate color palette
        colors = check_colors(colors, n_colors=len(groups), keys=groups,
                              palette=palette)

        # Get current axes if axes are not specified
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        # Times in the sample
        time = np.unique(self._data.time)

        estimate, lower, upper = self.predict(time, return_ci=True)

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

            # Plot the confidence bands
            if ci:
                ci_label = f"{self.conf_level:.0%} {self.conf_type} C.I."
                if len(groups) > 1:
                    ci_label += f" ({group})"
                c = p[0].get_color()
                alpha = 0.4 * curve_params.get("alpha", 1.)
                ci_params = dict(color=c, alpha=alpha, label=ci_label,
                                 step="post", zorder=2)
                if ci_kwargs is not None:
                    ci_params.update(ci_kwargs)
                ind = (~np.isnan(lower[group])) & (~np.isnan(upper[group]))
                ax.fill_between(time[ind], lower[group][ind], upper[group][ind],
                                **ci_params)

        # Configure axes
        ax.set(xlabel="time", ylabel=self._estimand)
        x_min, _ = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set(xlim=(max(x_min, 0), None), ylim=(min(y_min, 0), max(y_max, 1)))

        # Display the legend
        if legend and len(groups) > 1:
            legend_params = dict(loc="best", frameon=True, shadow=True)
            if legend_kwargs is not None:
                legend_params.update(legend_kwargs)
            ax.legend(**legend_params)

        return ax


class NonparametricSurvival(NonparametricEstimator):
    """Base class for nonparametric survival function estimators."""
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

    def quantile(self, prob):
        """Empirical quantile estimates for the time-to-event distribution.

        For a probability level :math:`p` between 0 and 1, the empirical
        :math:`p`-quantile of the time-to-event distribution with estimated
        survival function :math:`\hat{S}(t)` is defined to be the time at which
        the horizontal line at height :math:`1-p` intersects with the estimated
        survival curve. If such a time is not unique, then instead there is a
        time interval on which the estimated survival curve is flat and
        coincides with the horizontal line at height :math:`1-p`. In this case
        the midpoint of this interval is taken to be the empirical
        :math:`p`-quantile estimate (this is just one of many possible
        conventions, and the one used by the R package ``survival``). If the
        survival function estimate never gets as low as :math:`1-p`, then the
        :math:`p`-quantile cannot be estimated.

        Parameters
        ----------
        prob : array-like
            One-dimensional array of values between 0 and 1 representing the
            probability levels of the desired quantiles.

        Returns
        -------
        quantiles : pandas.DataFrame
            The quantile estimates. Rows are indexed by the entries of `time`
            and columns are indexed by the model's group labels. Entries for
            probability levels for which the quantile estimate is not defined
            are nan (not a number).
        """
        self.check_fitted()
        prob = check_data_1d(prob)
        if not np.all((prob >= 0) * (prob <= 1)):
            raise ValueError("Probability levels must be between zero and one.")

        quantiles = np.empty((prob.shape[0], self._data.n_groups),
                             dtype=np.float_)

        for j, g in enumerate(self._data.group_labels):
            # Find intersection of horizontal line at height 1-p with the
            # estimated survival curve
            cdf = np.concatenate(([0.], 1 - self.estimate_[j]))
            ind1 = np.searchsorted(cdf - self._quantile_tol, prob)
            ind2 = np.searchsorted(cdf + self._quantile_tol, prob)

            # Find out whether the last time for this group was censored
            censor_times = self._data.censor[g].time
            event_times = self._data.events[g].time
            if (len(censor_times) > 0
                    and censor_times.iloc[-1] > event_times.iloc[-1]):
                last = censor_times.iloc[-1]
            else:
                last = event_times.iloc[-1]

            qs = np.concatenate(([0.], event_times, [last]))
            quantiles[:, j] = 0.5 * (qs[ind1] + qs[ind2])

            # Special cases
            quantiles[prob < self._quantile_tol, j] = \
                np.min(self._data.entry[self._data.group == g])
            quantiles[prob > cdf[-1] + self._quantile_tol, j] = np.nan

        columns = pd.Index(self._data.group_labels, name="group")
        index = pd.Index(prob, name="prob")
        return pd.DataFrame(quantiles, columns=columns, index=index)


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
                * time
                    The distinct event times for the group.
                * events
                    Number of events at each distinct event time for the group.
                * at risk
                    Number of individuals at risk (i.e., entered but not yet
                    censored or failed) immediately before each distinct event
                    time for the group.
                * estimate
                    The estimate at each event time.
                * std. err.
                    The standard error of the estimate at each event time.
                * c.i. lower
                    Lower confidence interval bound for the estimate at each
                    event time. The actual name of this column will contain the
                    confidence level.
                * c.i. upper
                    Upper confidence interval bound for the estimate at each
                    event time. The actual name of this column will contain the
                    confidence level.
        """
        if group not in self.model.data_.group_labels:
            raise ValueError(f"Not a known group label: {group}.")

        # Group index
        i = (self.model.data_.group_labels == group).argmax()

        columns = ("time", "events", "at risk", "estimate", "std. error",
                   f"{self.model.conf_level:.0%} c.i. lower",
                   f"{self.model.conf_level:.0%} c.i. upper")

        time = self.model.data_.events[group].time
        events = self.model.data_.events[group].n_events
        at_risk = self.model.data_.events[group].n_at_risk

        estimate = self.model.estimate_[i]
        std_err = np.sqrt(self.model.estimate_var_[i])
        ci_lower = self.model.estimate_ci_lower_[i]
        ci_upper = self.model.estimate_ci_upper_[i]

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
