"""Implements storage of survival/lifetime data."""

import numpy as np
import pandas as pd

from survive.utils import check_data_1d


class SurvivalData(object):
    """Class representing right-censored survival/lifetime data.

    Properties
    ----------
    data : pandas.DataFrame
        A DataFrame containing the sample used to initialize this object.
        Columns:
            * time
                Each observed time (censored or not).
            * status
                Right-censoring indicators for each observed time. 0 indicates
                censoring, 1 indicates an event.
            * entry
                Entry times of the observations (for left truncation).
            * group
                Label for each observation's group/stratum within the sample.
    groups : numpy.ndarray
        List of the distinct groups in the sample.
    n_groups : int
        The number of distinct groups in the sample.
    time : numpy.ndarray of dtype object
        List of one-dimensional arrays, one for each group. Each array consists
        of the distinct event times (not censored) within that group.
    censor : numpy.ndarray of dtype object
        List of one-dimensional arrays, one for each group. Each array consists
        of the distinct times at which censoring occurred within that group.
    n_at_risk : numpy.ndarray of dtype object
        List of one-dimensional arrays, one for each group. Each array consists
        of the the size of the risk set (i.e., the number of individuals who
        have entered but have not yet experienced censoring or an event)
        immediately prior to each event time within that group.
    n_events : numpy.ndarray of dtype object
        List of one-dimensional arrays, one for each group. Each array consists
        of the the number of events occurring at each event time within that
        group.
    counts : pandas.DataFrame
        DataFrame containing the total number of observations, the number of
        true events, and the number of censored observations within each group.
        Columns:
            * observations
                The number of observations.
            * events
                The number of true events/failures.
            * censored
                The number of censored observations.
        Indices:
            The distinct group labels.
    """
    data: pd.DataFrame
    groups: np.ndarray
    n_groups: int
    time: np.ndarray
    censor: np.ndarray
    n_at_risk: np.ndarray
    n_events: np.ndarray
    counts: pd.DataFrame

    def __init__(self, time, *, status=None, entry=None, group=None, data=None):
        """Initialize a SurvivalData object.

        Parameters
        ----------
        time : one-dimensional array-like or str
            The observed times. If the DataFrame parameter `data` is provided,
            this can be the name of a column in `data` from which to get the
            observed times.
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
        """
        if data is not None:
            # In this case `data` must be a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Parameter 'data' must be a pandas DataFrame.")

            # Try to extract all the necessary information from the DataFrame
            time = _check_df_column(data, time)
            status = _check_df_column(data, status)
            entry = _check_df_column(data, entry)
            group = _check_df_column(data, group)

        # Validate observed times
        time = check_data_1d(time)
        if any(t < 0 for t in time):
            raise ValueError("Entries of 'time' must be non-negative.")

        # Validate censoring indicators
        if status is None:
            status = np.ones(time.shape[0], dtype=np.int_)
        else:
            status = check_data_1d(status, n_exact=time.shape[0], dtype=np.int_)
            if any(e not in (0, 1) for e in status):
                raise ValueError("Entries of 'status' must be 0 or 1.")

        # Validate entry times
        if entry is None:
            entry = np.zeros(time.shape[0], dtype=time.dtype)
        else:
            entry = check_data_1d(entry, n_exact=time.shape[0],
                                  dtype=time.dtype)
            if not all(0 <= t0 <= t for t0, t in zip(entry, time)):
                raise ValueError(
                    "Entries of 'entry' must be non-negative and at most as "
                    "large as the corresponding entries of 'time'.")

        # Validate group labels
        if group is None:
            group = np.zeros(time.shape[0], dtype=np.int_)
        else:
            group = check_data_1d(group, numeric=False, n_exact=time.shape[0])

        # Sort the times in increasing order, putting failures before censored
        # times in the case of ties
        ind = np.lexsort((1 - status, time), axis=0)
        time = time[ind]
        status = status[ind]
        entry = entry[ind]
        group = group[ind]

        # Put the initializer parameters into a DataFrame
        self.data = pd.DataFrame(dict(time=time, status=status, entry=entry,
                                      group=group))

        # Get distinct groups
        self.groups = np.unique(group)
        self.n_groups = self.groups.shape[0]

        # Compute survival quantities within each group and the rows of the
        # counts DataFrame
        self.time = np.empty(self.n_groups, dtype=object)
        self.censor = np.empty(self.n_groups, dtype=object)
        self.n_at_risk = np.empty(self.n_groups, dtype=object)
        self.n_events = np.empty(self.n_groups, dtype=object)
        rows = []
        for i, g in enumerate(self.groups):
            # Focus on one group at a time
            ind_g = (group == g)
            time_g = time[ind_g]
            status_g = status[ind_g]
            entry_g = entry[ind_g]

            # Get indices of true failures and censored observations
            ind_event = (status_g == 1)
            ind_censor = ~ind_event

            # Compute the current row of the counts DataFrame
            rows.append([time_g.shape[0], np.sum(ind_event),
                         np.sum(ind_censor)])

            # Distinct true event times and number of failures at such times
            self.time[i], self.n_events[i] \
                = np.unique(time_g[ind_event], return_counts=True)

            # Distinct censored observation times
            self.censor[i] = np.unique(time_g[ind_censor])

            # k = number of distinct observed times
            k = self.time[i].shape[0]

            # Number of individuals at risk immediately before each event time
            self.n_at_risk[i] = np.empty(shape=k, dtype=np.int_)
            for j in range(k):
                t = self.time[i][j]
                self.n_at_risk[i][j] = np.sum((entry_g < t) * (t <= time_g))

        # Put together the counts DataFrame
        columns = ("observations", "events", "censored")
        self.counts = pd.DataFrame(rows, columns=columns, index=self.groups)

    def __str__(self):
        """Short summary of the observation counts within each group."""
        return self.counts.to_string(index=True if self.n_groups > 1 else False)

    def table(self, group=None):
        """Get survival tables within groups.

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
            If no group is specified and there is more than one group total,
            then a list of such tables is returned (one for each group).
        """
        if group in self.groups:
            i = np.flatnonzero(self.groups == group)[0]
            return pd.DataFrame({"time": self.time[i],
                                 "at risk": self.n_at_risk[i],
                                 "events": self.n_events[i]})
        elif self.n_groups == 1:
            return self.table(group=self.groups[0])
        elif group is None:
            return [self.table(group=g) for g in self.groups]
        else:
            raise ValueError(f"Not a known group label: {group}.")


def _check_df_column(data: pd.DataFrame, name):
    """Check if `name` is the name of a column in a DataFrame `data`. If it is,
    return the column. Otherwise, return `name` unchanged."""
    if isinstance(name, str):
        if name in data.columns:
            return data[name]
        else:
            raise ValueError(f"Column '{name}' not found in DataFrame.")
    else:
        return name
