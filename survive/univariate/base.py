"""Abstract base class for univariate survival function estimators."""

import abc
import itertools

import numpy as np
import pandas as pd

from ..base import Model, Fittable, Predictor, SurvivalData, Summary
from ..utils.validation import check_float, check_data_1d


class UnivariateSurvival(Model, Fittable, Predictor):
    """Abstract base class for univariate survival function estimators.

    Properties
    ----------
    data : SurvivalData
        Survival data used to fit the estimator.
    conf_type : str
        Type of confidence intervals for the survival function estimate S(t) to
        report. Acceptable values depend on the actual estimator.
    conf_level : float
        Confidence level of the confidence intervals.
    """

    # Types of confidence intervals available
    _conf_types = (None,)

    # Storage of the survival data
    _data: SurvivalData

    # Internal versions of __init__() parameters
    _conf_type: str
    _conf_level: float

    @property
    def data(self) -> SurvivalData:
        """Survival data used to fit the estimator."""
        self.check_fitted()
        return self._data

    @property
    def conf_type(self):
        """Type of confidence intervals for the survival function to report."""
        return self._conf_type

    @conf_type.setter
    def conf_type(self, conf_type):
        """Set the type of confidence interval."""
        if self.fitted:
            raise RuntimeError("'conf_type' cannot be set after fitting.")
        elif conf_type in self._conf_types:
            self._conf_type = conf_type
        else:
            raise ValueError(f"Invalid value for 'conf_type': {conf_type}.")

    @property
    def conf_level(self):
        """Confidence level of the confidence intervals."""
        return self._conf_level

    @conf_level.setter
    def conf_level(self, conf_level):
        """Set the confidence level."""
        if self.fitted:
            raise RuntimeError("'conf_level' cannot be set after fitting.")
        self._conf_level = check_float(conf_level, minimum=0., maximum=1.)

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the univariate survival function estimator."""
        pass

    @abc.abstractmethod
    def _predict(self, *, t, i):
        """Get the univariate survival function estimates for a single group.

        Parameters
        ----------
        t : array-like
            One-dimensional array of non-negative times.
        i : int
            Index of the group whose survival function estimates should be
            returned.

        Returns
        -------
        prob : float or numpy.ndarray
            The survival function estimate. This is either a float or a
            one-dimensional array depending on whether the parameter `time` is a
            scalar or a one-dimensional array with at least two elements.
        """
        pass

    def predict(self, time, group=None):
        """Get the univariate survival function estimates.

        Parameters
        ----------
        time : array-like
            One-dimensional array of non-negative times.
        group : group label or None, optional (default: None)
            Specify the group whose survival function estimates should be
            returned. Ignored if there is only one group. If not specified,
            survival function estimates for all the groups are returned.

        Returns
        -------
        prob : float or numpy.ndarray or pandas.DataFrame
            The survival function estimate.
            Possible shapes:
                * If there is only one group or if a group is specified, then
                  this is either a float or a one-dimensional array depending on
                  whether the parameter `time` is a scalar or a one-dimensional
                  array with at least two elements, respectively.
                * If the model has more than one group and no group is
                  specified, then this is a pandas.DataFrame with as many rows
                  as entries in `time` and one column for each group.
        """
        self.check_fitted()
        time = check_data_1d(time)

        if group in self._data.groups:
            i = (self._data.groups == group).argmax()
            return self._predict(t=time, i=i)
        elif self._data.n_groups == 1:
            return self._predict(t=time, i=0)
        elif group is None:
            return pd.DataFrame({g: self._predict(t=time, i=i)
                                 for i, g in enumerate(self._data.groups)},
                                index=time)
        else:
            raise ValueError(f"Not a known group label: {group}.")

    @abc.abstractmethod
    def _var(self, *, t, i):
        """Estimate the variance of the estimated survival probability at the
        given times for a single group.

        Parameters
        ----------
        t : array-like
            One-dimensional array of non-negative times.
        i : int
            Index of the group whose survival function variance estimates should
            be returned.

        Returns
        -------
        var : float or numpy.ndarray
            The survival function variance estimate. This is either a float or a
            one-dimensional array depending on whether the parameter `time` is a
            scalar or a one-dimensional array with at least two elements.
        """
        pass

    def var(self, time, group=None):
        """Estimate the variance of the survival function estimates.

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
            The survival function variance estimate.
            Possible shapes:
                * If there is only one group or if a group is specified, then
                  this is either a float or a one-dimensional array depending on
                  whether the parameter `time` is a scalar or a one-dimensional
                  array with at least two elements, respectively.
                * If the model has more than one group and no group is
                  specified, then this is a pandas.DataFrame with as many rows
                  as entries in `time` and one column for each group.
        """
        self.check_fitted()
        time = check_data_1d(time)

        if group in self._data.groups:
            i = (self._data.groups == group).argmax()
            return self._var(t=time, i=i)
        elif self._data.n_groups == 1:
            return self._var(t=time, i=0)
        elif group is None:
            return pd.DataFrame({g: self._var(t=time, i=i)
                                 for i, g in enumerate(self._data.groups)},
                                index=time)
        else:
            raise ValueError(f"Not a known group label: {group}.")

    def se(self, time, group=None):
        """Estimate the standard error of the survival function estimates.

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
            The standard error estimates.
            Possible shapes:
                * If there is only one group or if a group is specified, then
                  this is either a float or a one-dimensional array depending on
                  whether the parameter `time` is a scalar or a one-dimensional
                  array with at least two elements, respectively.
                * If there is more than one group and no group is specified,
                  then this is a pandas.DataFrame with as many rows as entries
                  in `time` and one column for each group."""
        return np.sqrt(self.var(time, group=group))

    @abc.abstractmethod
    def _ci(self, *, t, i):
        """Confidence intervals for the estimated survival probability at the
        given times for a single group.

        Parameters
        ----------
        t : array-like
            One-dimensional array of non-negative times.
        i : int
            Index of the group whose survival function variance estimates should
            be returned.

        Returns
        -------
        lower : float or numpy.ndarray
        upper : float or numpy.ndarray
            Confidence interval upper and lower bounds. These are either floats
            or one-dimensional arrays depending on whether the parameter `time`
            is a scalar or a one-dimensional array with at least two elements.
        """
        pass

    def ci(self, time, group=None):
        """Confidence intervals for the survival function estimate.

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
        """
        self.check_fitted()
        time = check_data_1d(time)

        if group in self._data.groups:
            i = (self._data.groups == group).argmax()
            return self._ci(t=time, i=i)
        elif self._data.n_groups == 1:
            return self._ci(t=time, i=0)
        elif group is None:
            ls = np.empty(self._data.n_groups, dtype=object)
            us = np.empty(self._data.n_groups, dtype=object)
            for i, g in enumerate(self._data.groups):
                ls[i], us[i] = self._ci(t=time, i=i)
            lower = pd.DataFrame({g: l for g, l in zip(self._data.groups, ls)},
                                 index=time)
            upper = pd.DataFrame({g: u for g, u in zip(self._data.groups, us)},
                                 index=time)
            return lower, upper
        else:
            raise ValueError(f"Not a known group label: {group}.")

    @abc.abstractmethod
    def _quantile(self, *, p, i):
        """Estimates time-to-event distribution quantiles for a single group.

        Parameters
        ----------
        p : array-like
            One-dimensional array of probability levels.
        i : int
            Index of the group whose survival function variance estimates should
            be returned.

        Returns
        -------
        quantiles : float or numpy.ndarray
            The quantile estimates. This is either a float or a one-dimensional
            array depending on whether the parameter `time` is a scalar or a
            one-dimensional array with at least two elements.
        """
        pass

    def quantile(self, prob, group=None):
        """Quantile estimates for the time-to-event distribution.

        For a probability level p between 0 and 1, the p-quantile of the
        time-to-event distribution whose survival function S(t) is being
        estimated is defined to be the time at which the horizontal line at
        height 1-p intersects with the survival curve. If such a time is not
        unique, then instead there is a time interval on which the survival
        curve is flat and coincides with the horizontal line at height 1-p. In
        this case the midpoint of this interval is taken to be the p-quantile
        (this is just one of many possible conventions, and the one used by the
        R package ``survival``).

        Accordingly, estimates of the p-quantile from the survival function
        estimator are obtained by finding the time at the point of intersection
        of the survival curve estimate with the horizontal line at height 1-p.
        If there is an interval of such times, then its midpoint is selected (as
        described above). If the survival function estimate never gets as low as
        1-p, then the p-quantile cannot be estimated.

        Parameters
        ----------
        prob : array-like
            One-dimensional array of values between 0 and 1 representing the
            probability levels of the desired quantiles.
        group : group label or None, optional (default: None)
            Specify the group whose quantile estimates should be returned.
            Ignored if there is only one group. If not specified, quantile
            estimates for all the groups are returned.

        Returns
        -------
        quantiles : float or numpy.ndarray or pandas.DataFrame
            The quantiles.
            Possible shapes:
                * If there is only one group or if a group is specified, then
                  these are either floats or one-dimensional arrays depending
                  on whether the parameter `time` is a scalar or a
                  one-dimensional array with at least two elements,
                  respectively.
                * If there is more than one group and no group is specified,
                  then these are pandas.DataFrames with as many rows as entries
                  in `prob` and one column for each group.
            Entries for probability levels for which the quantile estimate is
            not defined are nan (not a number).

        Raises
        ------
        ValueError
            If an entry in `prob` is less than zero or greater than one.
        """
        self.check_fitted()

        # Validate parameters
        prob = check_data_1d(prob)
        if not np.all((prob >= 0) * (prob <= 1)):
            raise ValueError(
                "Probability levels must be between zero and one.")

        if group in self._data.groups:
            i = (self._data.groups == group).argmax()
            return self._quantile(p=prob, i=i)
        elif self._data.n_groups == 1:
            return self._quantile(p=prob, i=0)
        elif group is None:
            return pd.DataFrame({g: self._quantile(p=prob, i=i)
                                 for i, g in enumerate(self._data.groups)},
                                index=prob)
        else:
            raise ValueError(f"Not a known group label: {group}.")

    def describe(self):
        """Descriptive statistics about this survival function estimator.

        Returns
        -------
        table : pandas.DataFrame
            A DataFrame containing descriptive statistics for each group.
            Columns:
            * observations
                The number of observations.
            * events
                The number of true events/failures.
            * censored
                The number of censored observations.
            * 1st qu.
                The first quartile estimate (0.25-quantile).
            * median
                The median estimate (0.5-quantile).
            * 3rd qu.
                The third quartile estimate (0.75-quantile).
        Indices:
            The distinct group labels.
        """
        counts = self.data.counts
        qs = self.quantile([0.25, 0.5, 0.75])
        columns = ["1st qu.", "median", "3rd qu."]
        if self.data.n_groups == 1:
            # ``qs`` is a numpy.ndarray
            qs = pd.DataFrame({col: q for col, q in zip(columns, qs)},
                              index=counts.index)
        else:
            # ``qs`` is a pandas.DataFrame
            rows = qs.iterrows()
            qs = pd.DataFrame({col: q for col, (_, q) in zip(columns, rows)},
                              index=counts.index)
        return pd.concat((counts, qs), axis=1)

    def __str__(self):
        """Get the descriptive statistics table as a string."""
        return self.describe().to_string(index=(self.data.n_groups > 1))

    def summary(self):
        """Get a summary of this survival function estimator.

        Returns
        -------
        summary : SurvivalSummary
            The summary of this survival function estimator.
        """
        self.check_fitted()
        return UnivariateSurvivalSummary(self)


class NonparametricUnivariateSurvival(UnivariateSurvival):
    """Nonparametric survival function estimators.

    The resulting survival curves are step functions with jumps at the observed
    event times.
    """

    # Estimate of the survival function and related quantities at each observed
    # failure time within each group
    _survival: np.ndarray
    _survival_var: np.ndarray
    _survival_ci_lower: np.ndarray
    _survival_ci_upper: np.ndarray

    # Tolerance for checking if a probability level "exactly" equals a survival
    # probability when computing quantiles of the time-to-event distribution.
    # This is to counteract round-off error encountered when computing the
    # survival function estimates.
    _quantile_tol = np.sqrt(np.finfo(np.float_).eps)

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the nonparametric univariate survival function estimator."""
        pass

    def _predict(self, *, t, i):
        """Get the univariate survival function estimates for a single group.

        Parameters
        ----------
        t : array-like
            One-dimensional array of non-negative times.
        i : int
            Index of the group whose survival function estimates should be
            returned.

        Returns
        -------
        prob : float or numpy.ndarray
            The survival function estimate. This is either a float or a
            one-dimensional array depending on whether the parameter `time` is a
            scalar or a one-dimensional array with at least two elements.
        """
        ind = np.searchsorted(self._data.time[i], t, side="right")
        prob = np.concatenate(([1.], self._survival[i]))[ind]
        return prob.item() if prob.size == 1 else prob

    def _var(self, *, t, i):
        """Estimate the variance of the estimated survival probability at the
        given times for a single group.

        Parameters
        ----------
        t : array-like
            One-dimensional array of non-negative times.
        i : int
            Index of the group whose survival function variance estimates should
            be returned.

        Returns
        -------
        var : float or numpy.ndarray
            The survival function variance estimate. This is either a float or a
            one-dimensional array depending on whether the parameter `time` is a
            scalar or a one-dimensional array with at least two elements.
        """
        ind = np.searchsorted(self._data.time[i], t, side="right")
        var = np.concatenate(([0.], self._survival_var[i]))[ind]
        return var.item() if var.size == 1 else var

    def _ci(self, *, t, i):
        """Confidence intervals for the estimated survival probability at the
        given times for a single group.

        Parameters
        ----------
        t : array-like
            One-dimensional array of non-negative times.
        i : int
            Index of the group whose survival function variance estimates should
            be returned.

        Returns
        -------
        lower : float or numpy.ndarray
        upper : float or numpy.ndarray
            Confidence interval upper and lower bounds. These are either floats
            or one-dimensional arrays depending on whether the parameter `time`
            is a scalar or a one-dimensional array with at least two elements.
        """
        ind = np.searchsorted(self._data.time[i], t, side="right")
        lower = np.concatenate(([1.], self._survival_ci_lower[i]))[ind]
        upper = np.concatenate(([1.], self._survival_ci_upper[i]))[ind]
        return (lower.item() if lower.size == 1 else lower,
                upper.item() if upper.size == 1 else upper)

    def _quantile(self, *, p, i):
        """Estimates time-to-event distribution quantiles for a single group.

        Parameters
        ----------
        p : array-like
            One-dimensional array of probability levels.
        i : int
            Index of the group whose survival function variance estimates should
            be returned.

        Returns
        -------
        quantiles : float or numpy.ndarray
            The quantile estimates. This is either a float or a one-dimensional
            array depending on whether the parameter `time` is a scalar or a
            one-dimensional array with at least two elements.
        """
        cdf = np.concatenate(([0.], 1 - self._survival[i]))
        ind1 = np.searchsorted(cdf - self._quantile_tol, p)
        ind2 = np.searchsorted(cdf + self._quantile_tol, p)
        if (self.data.censor[i].shape[0] > 0
                and self.data.censor[i][-1] > self.data.time[i][-1]):
            last = self.data.censor[i][-1]
        else:
            last = self.data.time[i][-1]
        qs = np.concatenate(([0.], self.data.time[i], [last]))
        quantiles = 0.5 * (qs[ind1] + qs[ind2])

        # Special cases
        quantiles[p < self._quantile_tol] = np.min(self.data.data.entry)
        quantiles[p > cdf[-1] + self._quantile_tol] = np.nan

        return quantiles.item() if quantiles.size == 1 else quantiles

    def plot(self, *groups, ci=True, ci_kwargs=None, mark_censor=True,
             mark_censor_kwargs=None, legend=True, legend_kwargs=None,
             colors=None, palette=None, ax=None, **kwargs):
        """Plot the estimated survival curves.

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
        colors : list or tuple or dict or str, optional (default: None)
            Colors for each group's survival curve. This is ignored if `palette`
            is provided.
            Possible types:
                * list or tuple
                    Sequence of valid matplotlib colors to cycle through.
                * dict
                    Should be a dictionary with groups as keys and valid
                    matplotlib colors as values.
                * str
                    Name of a matplotlib colormap.
        palette : str, optional (default: None)
            Name of a seaborn color palette. Requires seaborn to be installed.
            Setting a color palette overrides the `colors` parameter.
        ax : matplotlib.axes.Axes, optional (default: None)
            The axes on which to draw the line. If this is not specified, the
            current axis will be used.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to step() when plotting
            survival curves.

        Returns
        -------
        The matplotlib.axes.Axes on which the curve was drawn.
        """
        self.check_fitted()

        if not groups:
            # Plot curves for all groups
            groups = self._data.groups
        else:
            # Ensure that the group names are valid
            for group in groups:
                if group not in self._data.groups:
                    raise ValueError(f"Not a known group label: {group}.")

        # Validate color palette
        if palette is not None:
            try:
                import seaborn as sns
            except ImportError:
                raise RuntimeError("The use of the 'palette' parameter "
                                   "requires seaborn to be installed.")
            colors = iter(sns.color_palette(palette, n_colors=len(groups)))
        elif isinstance(colors, list) or isinstance(colors, tuple):
            colors = itertools.cycle(colors)
        elif isinstance(colors, dict):
            for group in groups:
                if group not in colors:
                    raise ValueError(
                        f"Group {group} is not a key in dict 'colors'.")
            colors_copy = colors.copy()
            colors = (colors_copy[group] for group in groups)
        elif isinstance(colors, str):
            import matplotlib.pyplot as plt
            colormap = plt.get_cmap(colors)
            colors = iter(colormap(np.linspace(0., 1., len(groups))))
        elif colors is None:
            colors = itertools.repeat(None)
        else:
            raise ValueError(f"Invalid value for parameter 'colors': {colors}.")

        # Get current axes if axes are not specified
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        # Plot the survival curves
        for group in groups:
            # Group index
            i = (self._data.groups == group).argmax()

            # Unique times in the sample
            x = np.unique(np.concatenate((self._data.time[i],
                                          self._data.censor[i])))

            # Survival probabilities
            y = self.predict(x, group=group)

            # Parameters for the survival curve plot
            color = next(colors)
            curve_label = f"{self.model_type}"
            if len(groups) > 1:
                curve_label += f" ({group})"
            curve_params = dict(where="post", label=curve_label, zorder=3)
            if color is not None:
                curve_params["color"] = color
            curve_params.update(kwargs)

            # Plot this group's survival curve
            p = ax.step(x, y, **curve_params)

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
            if ci and self._conf_type is not None:
                lower, upper = self.ci(x, group=group)
                ci_label = f"{self.conf_level:.0%} {self.conf_type} C.I."
                if len(groups) > 1:
                    ci_label += f" ({group})"
                c = p[0].get_color()
                alpha = 0.4 * curve_params.get("alpha", 1.)
                ci_params = dict(color=c, alpha=alpha, label=ci_label,
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


class UnivariateSurvivalSummary(Summary):
    """Summaries for survival function estimators.

    Properties
    ----------
    model : UnivariateSurvival
        The survival function estimator being summarized.
    """
    model: UnivariateSurvival

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
            If confidence interval bounds should be returned, then there are
            two more columns.
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
            i = (self.model.data.groups == group).argmax()
            columns = ("survival", "std. err.")
            survival = self.model.predict(self.model.data.time[i], group=group)
            se = self.model.se(self.model.data.time[i], group=group)
            data = (survival, se)
            if self.model.conf_type is not None:
                columns += (f"{self.model.conf_level:.0%} c.i. lower",
                            f"{self.model.conf_level:.0%} c.i. upper")
                data += self.model.ci(self.model.data.time[i], group=group)
            table1 = self.model.data.table(group=group)
            table2 = pd.DataFrame(dict(zip(columns, data)))
            return pd.concat((table1, table2), axis=1)
        elif self.model.data.n_groups == 1:
            return self.table(group=self.model.data.groups[0])
        elif group is None:
            return [self.table(group=g) for g in self.model.data.groups]
        else:
            raise ValueError(f"Not a known group label: {group}.")

    def __str__(self):
        """Return a string summary of the survivor function estimator."""
        describe = self.model.describe()
        summary = super(UnivariateSurvivalSummary, self).__str__()
        for group in self.model.data.groups:
            if self.model.data.n_groups > 1:
                summary += f"\n\n{group}"
            summary += f"\n\n{describe.loc[[group]].to_string(index=False)}"
            summary += f"\n\n{self.table(group).to_string(index=False)}"
        return summary
