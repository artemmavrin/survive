"""Implements storage of survival/lifetime data."""

import warnings

import numpy as np
import pandas as pd

from .utils import check_data_1d, check_colors, check_float, check_bool

# Defaults for SurvivalData string formatting
_MAX_LINE_LENGTH = 75
_SEPARATOR = " "
_CENSOR_MARKER = "+"
_DEFAULT_FORMATTING = dict(max_line_length=_MAX_LINE_LENGTH,
                           separator=_SEPARATOR,
                           censor_marker=_CENSOR_MARKER)


class SurvivalData(object):
    """Class representing right-censored and left-truncated survival data.

    Parameters
    ----------
    time : array-like or str
        The observed times. If the DataFrame parameter `df` is provided, this
        can be the name of a column in `df` from which to get the observed
        times. Otherwise this should be a one-dimensional array of positive
        numbers.

    status : array-like or str, optional
        Censoring indicators. 0 means a right-censored observation, 1 means a
        true failure/event. If not provided, it is assumed that there is no
        censoring.  If the DataFrame parameter `df` is provided, this can be
        the name of a column in `df` from which to get the censoring indicators.
        Otherwise this should be an array of 0's and 1's of the same shape as
        the array of observed times.

    entry : array-like or str, optional
        Entry/birth times of the observations (for left-truncated data). If not
        provided, the entry time for each observation is set to 0. If the
        DataFrame parameter `df` is provided, this can be the name of a column
        in `df` from which to get the entry times. Otherwise this should be an
        array of non-negative numbers of the same shape as the array of observed
        times.

    group : array-like or string, optional
        Group/stratum labels for each observation. If not provided, the entire
        sample is taken as a single group. If the DataFrame parameter `df`
        is provided, this can be the name of a column in `df` from which to
        get the group labels. Otherwise this should be an array of the same
        shape as the array of observed times.

    df : pandas.DataFrame, optional
        Optional :class:`pandas.DataFrame` from which to extract the data. If
        this parameter is specified, then the parameters `time`, `status`,
        `entry`, and `group` can be column names of this DataFrame.

    min_time : numeric, optional
        The minimum observed time to consider part of the sample. This is for
        conditional inference. Observations with earlier observed event or
        censoring times are ignored. If not provided, all observations are used.

    warn : bool, optional
        Indicates whether any warnings should be raised or ignored (e.g., if an
        individual's entry time is later than that individual's event time).

    Attributes
    ----------
    time : numpy.ndarray
        Each observed time.

    status : numpy.ndarray
        Event indicators for each observed time. 1 indicates an event, 0
        indicates censoring.

    entry : numpy.ndarray
        Entry times of the observations (for left truncation).

    group : numpy.ndarray
        Label for each observation's group/stratum within the sample.

    group_labels : numpy.ndarray
        List of the distinct groups in the sample.

    n_groups : int
        The number of distinct groups in the sample.

    events : dict
        Mapping of group labels to DataFrames with columns:
            * time
                Distinct event times for that group
            * n_events
                Number of events at each event time.
            * n_at_risk
                Number of individuals at risk at each event time.

    censor : dict
        Mapping of group labels to DataFrames with columns:
            * time
                Distinct censored times for that group
            * n_censor
                Number of individuals censored at each censored time.
            * n_at_risk
                Number of individuals at risk at each censored time.
    """
    time: np.ndarray
    status: np.ndarray
    entry: np.ndarray
    group: np.ndarray
    group_labels: np.ndarray
    n_groups: int
    events: dict
    censor: dict

    # Dictionary of string formatting options (used by the to_string() and
    # __repr__() methods)
    _formatting: dict

    def __init__(self, time, *, status=None, entry=None, group=None, df=None,
                 min_time=None, warn=True):
        # Validate parameters
        warn = check_bool(warn)
        min_time = check_float(min_time, allow_none=True)

        if df is not None:
            # In this case df must be a DataFrame
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Parameter 'df' must be a pandas DataFrame.")

            # Try to extract all the necessary information from the DataFrame
            time = _check_df_column(df, time)
            status = _check_df_column(df, status)
            entry = _check_df_column(df, entry)
            group = _check_df_column(df, group)

        # Validate observed times
        time = check_data_1d(time, keep_pandas=False)
        n = time.shape[0]

        # Validate censoring indicators
        if status is None:
            status = np.ones(time.shape[0], dtype=np.int_)
        else:
            status = check_data_1d(status, n_exact=n, keep_pandas=False)
            if any(e not in (0, 1) for e in status):
                raise ValueError("Entries of 'status' must be 0 or 1.")

        # Validate entry times
        if entry is None:
            entry = np.zeros(time.shape[0], dtype=time.dtype)
        else:
            entry = check_data_1d(entry, n_exact=n, keep_pandas=False)

        # Validate group labels
        if group is None:
            # The entire sample is a single group labelled 0
            group = np.zeros(time.shape[0], dtype=np.int_)
        else:
            group = check_data_1d(group, numeric=False, n_exact=n,
                                  keep_pandas=False)

        # Ignore times before the minimum time
        if min_time is not None:
            mask = (time >= min_time)
            time = time[mask]
            status = status[mask]
            entry = entry[mask]
            group = group[mask]

        # Remove data where entry time is at or later than the observed time
        mask = (time > entry)
        if warn and not np.all(mask):
            warnings.warn(
                f"Ignoring {np.sum(~mask)} observations where entry >= time.")
        time = time[mask]
        status = status[mask]
        entry = entry[mask]
        group = group[mask]

        # Sort the times in increasing order, putting failures before censored
        # times in the case of ties. This is because we assume that censored
        # individuals do not die immediately after being censored.
        sort_indices = np.lexsort((1 - status, time), axis=0)
        self.time = np.array(time[sort_indices])
        self.status = np.array(status[sort_indices])
        self.entry = np.array(entry[sort_indices])
        self.group = np.array(group[sort_indices])

        # Extract distinct group labels
        self.group_labels = np.unique(self.group)
        self.n_groups = self.group_labels.shape[0]

        # Get the distinct event and censored times for each group, along with
        # the number of events/censored times and number of individuals at risk
        # at each time
        self.events = dict()
        self.censor = dict()
        mask_events = (self.status == 1)
        mask_censor = ~mask_events
        columns_events = pd.Index(("time", "n_events", "n_at_risk"))
        columns_censor = pd.Index(("time", "n_censor", "n_at_risk"))
        for grp in self.group_labels:
            mask_group = (self.group == grp)
            t0 = self.entry[mask_group]
            t1 = self.time[mask_group]

            # Event times
            e_times, e_counts = np.unique(self.time[mask_events & mask_group],
                                          return_counts=True)
            e_at_risk = _n_at_risk(time=e_times, t0=t0, t1=t1)
            events_dict = dict(zip(columns_events,
                                   (e_times, e_counts, e_at_risk)))

            # Censored times
            c_times, c_counts = np.unique(self.time[mask_censor & mask_group],
                                          return_counts=True)
            c_at_risk = _n_at_risk(time=c_times, t0=t0, t1=t1)
            censor_dict = dict(zip(columns_censor,
                                   (c_times, c_counts, c_at_risk)))

            self.events[grp] = pd.DataFrame(events_dict)
            self.censor[grp] = pd.DataFrame(censor_dict)

        # Set default string formatting options
        self.reset_format()

    def set_format(self, **kwargs):
        """Set string formatting options.

        Parameters
        ----------
        **kwargs : keyword arguments
            Formatting options. Allowed arguments:
                * max_line_length : int
                    Specify the maximum length of a single line.
                * separator : str
                    Specify how to separate individual times.
                * censor_marker : str
                    String to mark censored times.
        """
        for k, v in kwargs.items():
            if k in _DEFAULT_FORMATTING.keys():
                self._formatting[k] = v
            else:
                raise RuntimeError(f"Unknown formatting option: {k}.")

    def reset_format(self):
        """Restore string formatting defaults."""
        self._formatting = _DEFAULT_FORMATTING.copy()

    def to_string(self, group=None, *, max_line_length=None, separator=None,
                  censor_marker=None):
        """Get a string representation of the survival data within a group.

        Parameters
        ----------
        group : group label, optional
            Specify a single group to represent. If no group is specified, then
            the entire sample is treated as one group.

        max_line_length : int, optional
            Specify the maximum length of a single line.

        separator : str, optional
            Specify how to separate individual times.

        censor_marker : str, optional
            String to mark censored times.

        Returns
        -------
        str
            String representation of the observed survival times within a group.
        """
        # Get either times and censoring indicators for a single group or for
        # the whole sample
        if group in self.group_labels:
            mask = (self.group == group)
            time = self.time[mask]
            status = self.status[mask]
        elif group is None:
            time = self.time
            status = self.status
        else:
            raise ValueError(f"Not a known group label: {group}.")

        # Get default formatting options if not provided
        if max_line_length is None:
            max_line_length = self._formatting["max_line_length"]
        if separator is None:
            separator = self._formatting["separator"]
        if censor_marker is None:
            censor_marker = self._formatting["censor_marker"]

        # Pad event times with spaces on the right so that the event and
        # censored times line up
        event_marker = " " * len(censor_marker)

        # Get each observed time as a string together with censoring markers
        if np.any(status == 0):
            time_str = [str(t) + (event_marker if d else censor_marker)
                        for t, d in zip(time, status)]
        else:
            time_str = list(map(str, time))

        # Padding for each time
        pad = max(map(len, time_str))

        this_line_length = 0
        strings = []
        for i, t in enumerate(time_str):
            this_string = f"{t:>{pad}}"
            if i > 0:
                if this_line_length + len(this_string) >= max_line_length:
                    # Start a new line
                    strings.append(separator.rstrip() + "\n")
                    this_line_length = 0
                else:
                    # Continue current line
                    strings.append(separator)
                    this_line_length += len(separator)
            strings.append(this_string)
            this_line_length += len(this_string)

        return "".join(strings)

    def __repr__(self):
        """Get a string representation of the survival data."""
        strings = []
        for i, group in enumerate(self.group_labels):
            if i > 0:
                strings.append("\n\n")
            if self.n_groups > 1:
                strings.append(f"{group}\n\n")
            strings.append(self.to_string(group=group))

        return "".join(strings)

    @property
    def describe(self):
        """Get a DataFrame with descriptive statistics about the survival data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with a row for every group. The columns are
                * total
                    The total number of observations within a group
                * events
                    The number of events within a group
                * censored
                    The number of censored events within a group
        """
        total = np.empty(self.n_groups, dtype=np.int_)
        events = np.empty(self.n_groups, dtype=np.int_)

        for i, group in enumerate(self.group_labels):
            mask = (self.group == group)
            total[i] = np.sum(mask)
            events[i] = np.sum(mask & (self.status == 1))

        censored = total - events

        columns = ["total", "events", "censored"]
        index = pd.Index(self.group_labels, name="group")
        describe_dict = dict(zip(columns, (total, events, censored)))
        return pd.DataFrame(describe_dict, index=index)

    def n_at_risk(self, time):
        """Get the number of individuals at risk (i.e., entered but yet to
        undergo an event or censoring) at the given times.

        Parameters
        ----------
        time : float or array-like
            Times at which to report the risk set sizes.

        Returns
        -------
        pandas.DataFrame
            Number of individuals at risk at the given times within each group.
            The rows are indexed by the times in `time`, and the columns are
            indexed by group.
        """
        # Validate array of times
        time = check_data_1d(time)

        # Compute the risk set sizes at the given times within each group
        n_at_risk = np.empty(shape=(len(time), self.n_groups), dtype=np.int_)
        for col, group in enumerate(self.group_labels):
            mask = (self.group == group)
            t0 = self.entry[mask]
            t1 = self.time[mask]
            n_at_risk[:, col] = _n_at_risk(time=time, t0=t0, t1=t1)

        columns = pd.Index(self.group_labels, name="group")
        index = pd.Index(time, name="time")
        return pd.DataFrame(n_at_risk, columns=columns, index=index)

    def n_events(self, time):
        """Get the number of events at the given times.

        Parameters
        ----------
        time : float or array-like
            Times at which to report the numbers of events.

        Returns
        -------
        pandas.DataFrame
            Number of events at the given times within each group. The rows are
            indexed by the times in `time`, and the columns are indexed by
            group.
        """
        # Validate array of times
        time = check_data_1d(time)

        # Initialize array of event counts (to be converted to a DataFrame
        # later)
        n_events = np.empty(shape=(len(time), self.n_groups), dtype=np.int_)

        # Compute the event counts at the given times within each group
        mask = (self.status == 1)
        for col, group in enumerate(self.group_labels):
            event_time = self.time[mask & (self.group == group)]
            for row, t in enumerate(time):
                n_events[row, col] = np.sum(event_time == t)

        columns = pd.Index(self.group_labels, name="group")
        index = pd.Index(time, name="time")
        return pd.DataFrame(n_events, columns=columns, index=index)

    def plot_lifetimes(self, legend=True, legend_kwargs=None, colors=None,
                       palette=None, ax=None, **kwargs):
        """Plot the observed survival times.

        Parameters
        ----------
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
            :func:`matplotlib.axes.Axes.plot` when plotting the lifetimes.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the plot was drawn.
        """
        # Validate colors
        colors = check_colors(colors, n_colors=len(self.group_labels),
                              keys=self.group_labels, palette=palette)

        # Sort data by group, then by entry, then by observed time, putting
        # events before censored times
        ind = np.lexsort((1 - self.status, self.time, self.entry, self.group),
                         axis=0)
        time = self.time[ind]
        status = self.status[ind]
        entry = self.entry[ind]
        group = self.group[ind]

        # Get current axes if axes are not specified
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        colors = dict(zip(self.group_labels, colors))

        # Plot the data
        groups_seen = set()
        for i, (t0, t1, d, g) in enumerate(zip(entry, time, status, group)):
            # Parameters for the plot
            color = colors[g]
            params = dict()
            if color is not None:
                params["color"] = color
            if g not in groups_seen:
                params["label"] = str(g)
                groups_seen.add(g)
            params.update(kwargs)

            p = ax.plot([t0, t1], [i, i], **params)
            if d == 0:
                ax.plot(t1, i, marker="o", markersize=5, color=p[0].get_color())

            if color is None:
                colors[g] = p[0].get_color()

        # Configure axes
        ax.set(xlabel="time")
        x_min, _ = ax.get_xlim()
        y_offset = 2
        ax.set(xlim=(max(x_min, 0), None))
        ax.set(ylim=(-y_offset, time.shape[0] + y_offset - 1))
        ax.get_yaxis().set_visible(False)

        # Display the legend
        if legend:
            legend_params = dict(loc="best", frameon=True, shadow=True)
            if legend_kwargs is not None:
                legend_params.update(legend_kwargs)
            ax.legend(**legend_params)

        return ax

    def plot_at_risk(self, legend=True, legend_kwargs=None, colors=None,
                     palette=None, ax=None, **kwargs):
        """Plot the at-risk process.

        Parameters
        ----------
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
            :func:`matplotlib.axes.Axes.step` when plotting the at-risk process.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the plot was drawn.
        """
        # If there is left truncation, include entry times
        if np.any(self.entry > 0):
            time = np.unique(np.concatenate((self.time, self.entry)))
        else:
            time = np.unique(self.time)
        y = self.n_at_risk(time)

        colors = check_colors(colors, n_colors=len(self.group_labels),
                              keys=self.group_labels, palette=palette)

        # Get current axes if axes are not specified
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        # Plot the at-risk counts
        for group in self.group_labels:
            # Parameters for the plot
            color = next(colors)
            label = f"{group}"
            params = dict(where="post", label=label)
            if color is not None:
                params["color"] = color
            params.update(kwargs)

            # Plot this group's survival curve
            ax.step(time, y[group], **params)

        # Configure axes
        ax.set(xlabel="time", ylabel="number at risk")
        ax.autoscale(enable=True, axis="x")
        x_min, _ = ax.get_xlim()
        y_min, _ = ax.get_ylim()
        ax.set(xlim=(max(x_min, 0), None), ylim=(min(y_min, 0), None))

        # Display the legend
        if legend:
            legend_params = dict(loc="best", frameon=True, shadow=True)
            if legend_kwargs is not None:
                legend_params.update(legend_kwargs)
            ax.legend(**legend_params)

        return ax


def _check_df_column(df: pd.DataFrame, name):
    """Check if `name` is the name of a column in a DataFrame `df`. If it is,
    return the column. Otherwise, return `name` unchanged.
    """
    if isinstance(name, str):
        if name in df.columns:
            return df[name]
        else:
            raise ValueError(f"Column '{name}' not found in DataFrame.")
    else:
        return name


def _n_at_risk(time, t0, t1):
    """Compute number of individuals at risk at each time in `time`. The entry
    and exit times of the data are `t0` and `t1`, respectively.
    """
    n_at_risk = np.empty(len(time), dtype=np.int_)
    for i, t in enumerate(time):
        n_at_risk[i] = np.sum((t0 < t) & (t <= t1))
    return n_at_risk
