"""Implements storage of survival/lifetime data."""

import numpy as np
import pandas as pd

from survive.utils import check_data_1d


class SurvivalData(object):
    """Class representing right-censored survival/lifetime data.

    Properties
    ----------
    data : pandas.DataFrame
        A DataFrame containing the data used to initialize this object.
        Columns:
            * Time
                Each observed time-to-event.
            * Event
                Right-censoring indicators for each time-to-event. 0 indicates
                censoring, 1 indicates a true event.
            * Entry
                Entry times of the observations (left truncation).
    time : numpy.ndarray of shape (k,)
        One-dimensional array of the k distinct observed times.
    n_at_risk : numpy.ndarray of shape (k,)
        Counts of the number of individuals at risk (i.e., yet to undergo
        failure or censoring) among those who have entered before the current
        time-to-event in `time`.
    n_fail : numpy.ndarray of shape (k,)
        Counts of the number of true failures at the current time-to-event in
        `time`.
    n_censor : numpy.ndarray
        Counts of the number of censored events at the current time-to-event in
        `time`.
    table : pandas.DataFrame
        Summary table of the data.
        Columns:
            * Time
                The distinct observed times.
            * At Risk
                Number of individuals at risk at each distinct observed time.
            * Fail
                Number of failures/true events at each distinct observed time.
            * Censor
                Number of censored events at each distinct observed time.
    """
    data: pd.DataFrame
    time: np.ndarray
    n_at_risk: np.ndarray
    n_fail: np.ndarray
    n_censor: np.ndarray
    table: pd.DataFrame

    def __init__(self, time, event=None, entry=None):
        """Initialize a SurvivalData
         object.

        Parameters
        ----------
        time : array-like, one-dimensional
            The observed times.
        event : array-like, one-dimensional, optional (default: None)
            Censoring indicators. 0 means a right-censored observation, 1 means
            a true failure/event. If not provided, it is assumed that there is
            no censoring.
        entry : array-like, one-dimensional, optional (default: None)
            Entry times of the observations (for left-truncated data). If not
            provided, the entry time for each observation is set to 0.
        """
        # Validate observed times
        time = check_data_1d(time)
        if any(t < 0 for t in time):
            raise ValueError("Entries of 'time' must be non-negative.")

        # Validate censoring indicators
        if event is None:
            event = np.ones(time.shape[0], dtype=np.int_)
        else:
            event = check_data_1d(event, n_exact=time.shape[0], dtype=np.int_)
            if any(e not in (0, 1) for e in event):
                raise ValueError("Entries of 'event' must be 0 or 1.")

        # Validate entry times
        if entry is None:
            entry = np.zeros(time.shape[0], dtype=time.dtype)
        else:
            entry = check_data_1d(entry, n_exact=time.shape[0],
                                  dtype=time.dtype)
            if not all(0 <= q <= t for q, t in zip(entry, time)):
                raise ValueError("Entries of 'entry' must be non-negative and "
                                 "at most as large as the corresponding "
                                 "entries of 'time'.")

        # Sort the times in increasing order, putting failures before censored
        # times in the case of ties
        ind = np.lexsort((1 - event, time), axis=0)
        time = time[ind]
        event = event[ind]
        entry = entry[ind]

        # Distinct observed times
        self.time = np.unique(time)

        # k = number of distinct observed times
        k = self.time.shape[0]

        # Number of individuals at risk immediately before each observed time,
        # true failures, and censored events at each observed time
        self.n_at_risk = np.empty(shape=k, dtype=np.int_)
        self.n_fail = np.empty(shape=k, dtype=np.int_)
        self.n_censor = np.empty(shape=k, dtype=np.int_)
        mask_fail = (event == 1)
        mask_censor = (event == 0)
        for i in range(k):
            t = self.time[i]
            self.n_at_risk[i] = np.sum((entry < t) * (t <= time))
            mask_time = (time == t)
            self.n_fail[i] = np.sum(mask_fail * mask_time)
            self.n_censor[i] = np.sum(mask_censor * mask_time)

        # Put the initializer parameters into a DataFrame
        columns = ("Time", "Event", "Entry")
        self.data = pd.DataFrame(dict(zip(columns, (time, event, entry))))

        # Create summary table
        columns = ("Time", "At Risk", "Fail", "Censor")
        summary = (self.time, self.n_at_risk, self.n_fail, self.n_censor)
        self.table = pd.DataFrame(dict(zip(columns, summary)))

    def __str__(self):
        """String representation of the survival data."""
        return self.table.to_string(index=False)
