"""Implements storage of survival/lifetime data."""

import numpy as np
import pandas as pd

from ..utils import check_data_1d, check_colors


class SurvivalData(object):
    """Class representing right-censored and left-truncated survival data.

    Properties
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
    """

    def __init__(self, time, *, status=None, entry=None, group=None, df=None):
        """Initialize a SurvivalData object.

        Parameters
        ----------
        time : one-dimensional array-like or str
            The observed times. If the DataFrame parameter ``df`` is provided,
            this can be the name of a column in ``df`` from which to get the
            observed times.
        status : one-dimensional array-like or str, optional (default: None)
            Censoring indicators. 0 means a right-censored observation, 1 means
            a true failure/event. If not provided, it is assumed that there is
            no censoring.  If the DataFrame parameter ``df`` is provided,
            this can be the name of a column in ``df`` from which to get the
            censoring indicators.
        entry : one-dimensional array-like or str, optional (default: None)
            Entry/birth times of the observations (for left-truncated data). If
            not provided, the entry time for each observation is set to 0. If
            the DataFrame parameter ``df`` is provided, this can be the name of
            a column in ``df`` from which to get the entry times.
        group : one-dimensional array-like or string, optional (default: None)
            Group/stratum labels for each observation. If not provided, the
            entire sample is taken as a single group. If the DataFrame parameter
            ``df`` is provided, this can be the name of a column in ``df`` from
            which to get the group labels.
        df : pandas.DataFrame, optional (default: None)
            Optional DataFrame from which to extract the data. If this parameter
            is specified, then the parameters ``time``, ``status``, ``entry``,
            and ``group`` can be column names of this DataFrame.
        """
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
        if any(t < 0 for t in time):
            raise ValueError("Entries of 'time' must be non-negative.")

        # Validate censoring indicators
        if status is None:
            status = np.ones(time.shape[0], dtype=np.int_)
        else:
            status = check_data_1d(status, n_exact=time.shape[0],
                                   keep_pandas=False, dtype=np.int_)
            if any(e not in (0, 1) for e in status):
                raise ValueError("Entries of 'status' must be 0 or 1.")

        # Validate entry times
        if entry is None:
            entry = np.zeros(time.shape[0], dtype=time.dtype)
        else:
            entry = check_data_1d(entry, n_exact=time.shape[0],
                                  keep_pandas=False, dtype=time.dtype)

        # Validate group labels
        if group is None:
            group = np.zeros(time.shape[0], dtype=np.int_)
        else:
            group = check_data_1d(group, numeric=False, n_exact=time.shape[0],
                                  keep_pandas=False)

        # Remove data where entry is later than the observed time (should this
        # raise a warning?)
        ind = (time >= entry)
        time = time[ind]
        status = status[ind]
        entry = entry[ind]
        group = group[ind]

        # Sort the times in increasing order, putting failures before censored
        # times in the case of ties. This is because we assume that censored
        # individuals do not die immediately after being censored.
        ind = np.lexsort((1 - status, time), axis=0)
        self.time = time[ind]
        self.status = status[ind]
        self.entry = entry[ind]
        self.group = group[ind]

        self.group_labels = np.unique(self.group)
        self.n_groups = self.group_labels.shape[0]

    def to_string(self, group=None, max_line_length=None, separator=None,
                  censor_marker=None):
        """Get a string representation of the survival data within each group.

        Parameters
        ----------
        group : group label, optional (default: None)
            Specify a single group to represent. If no group is specified, then
            the entire sample is treated as one group.
        max_line_length : int, optional (default: None)
            Specify the maximum length of a single line.
        separator : str, optional (default: None)
            Specify how to separate individual times.
        censor_marker : str, optional (default: None)
            String to mark censored times.

        Returns
        -------
        string : str or list of strings
            String representation of the observed survival times within a group
            or within each group.
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

        # Default formatting options
        if max_line_length is None:
            max_line_length = 80
        if separator is None:
            separator = " "
        if censor_marker is None:
            censor_marker = "+"

        # Pad event times with spaces on the right so that the event and
        # censored times line up
        event_marker = " " * len(censor_marker)

        # Each observed time as a string together with censoring markers
        if np.any(status == 0):
            # There is censoring so censoring markers are needed
            time_str = [str(t) + (event_marker if d else censor_marker)
                        for t, d in zip(time, status)]
        else:
            # There is no censoring so censoring markers are not needed
            time_str = list(map(str, time))

        # Padding for each time
        pad = max(map(len, time_str))

        this_line_length = 0
        string = ""
        for i, t in enumerate(time_str):
            this_string = f"{t:>{pad}}"
            if i > 0:
                if this_line_length + len(this_string) >= max_line_length:
                    # Start a new line
                    string += separator.rstrip() + "\n"
                    this_line_length = 0
                else:
                    # Continue current line
                    string += separator
                    this_line_length += len(separator)
            string += this_string
            this_line_length += len(this_string)

        return string

    def __repr__(self):
        """Get a string representation of all the survival data."""
        survival_repr = ""
        for i, group in enumerate(self.group_labels):
            if i > 0:
                survival_repr += "\n\n"
            if self.n_groups > 1:
                survival_repr += f"{group}\n\n"
            survival_repr += self.to_string(group=group)

        return survival_repr

    def n_at_risk(self, time):
        """Get the number of individuals at risk (i.e., entered but yet to
        undergo an event or censoring) at the given times.

        Parameters
        ----------
        time : float or array-like
            Times at which to report the risk set sizes.

        Returns
        -------
        n_at_risk : pandas.DataFrame
            Number of individuals at risk at the given times within each group.
            The rows are indexed by the times in ``time``, and the columns are
            indexed by group.
        """
        # Validate array of times
        time = check_data_1d(time)

        # Compute the risk set sizes at the given times within each group
        n_at_risk = np.empty(shape=(len(time), self.n_groups), dtype=np.int_)
        for col, group in enumerate(self.group_labels):
            mask = (self.group == group)
            # t0 = entry time, t1 = exit time
            t0 = self.entry[mask]
            t1 = self.time[mask]
            for row, t in enumerate(time):
                # An individual is "at risk" at time t if their entry time t0 is
                # at or before t and their exit time t1 is at or after t
                n_at_risk[row, col] = np.sum((t0 <= t) & (t <= t1))

        columns = pd.Index(self.group_labels, name="group")
        index = pd.Index(time, name="time")
        return pd.DataFrame(n_at_risk, columns=columns, index=index)

    def n_events(self, time):
        """Get the number of events at the given times.

        Parameters
        ----------
        time : float or array-like
            Times at which to report the risk set sizes.

        Returns
        -------
        n_events : pandas.DataFrame
            Number of events at the given times within each group. The rows are
            indexed by the times in ``time``, and the columns are indexed by
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
        legend : bool, optional (default: True)
            Indicates whether to display a legend for the plot.
        legend_kwargs : dict, optional (default: None)
            Keyword parameters to pass to legend().
        colors : list or tuple or dict or str, optional (default: None)
            Colors for each group. This is ignored if ``palette`` is provided.
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
            Setting a color palette overrides the ``colors`` parameter.
        ax : matplotlib.axes.Axes, optional (default: None)
            The axes on which to plot. If this is not specified, the current
            axis will be used.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to the plot() function.

        Returns
        -------
        The matplotlib.axes.Axes on which the plot was drawn.
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
        legend : bool, optional (default: True)
            Indicates whether to display a legend for the plot.
        legend_kwargs : dict, optional (default: None)
            Keyword parameters to pass to legend().
        colors : list or tuple or dict or str, optional (default: None)
            Colors for each group. This is ignored if ``palette`` is provided.
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
            Setting a color palette overrides the ``colors`` parameter.
        ax : matplotlib.axes.Axes, optional (default: None)
            The axes on which to plot. If this is not specified, the current
            axis will be used.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to step() when plotting.

        Returns
        -------
        The matplotlib.axes.Axes on which the plot was drawn.
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

    @property
    def describe(self):
        """Get a DataFrame with descriptive statistics about the survival data.

        Returns
        -------
        describe : pandas.DataFrame
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


def _check_df_column(df: pd.DataFrame, name):
    """Check if ``name`` is the name of a column in a DataFrame ``df``.
    If it is, return the column. Otherwise, return ``name`` unchanged.
    """
    if isinstance(name, str):
        if name in df.columns:
            return df[name]
        else:
            raise ValueError(f"Column '{name}' not found in DataFrame.")
    else:
        return name
