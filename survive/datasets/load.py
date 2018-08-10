"""Functions for loading the packaged datasets."""

import os
import pathlib

import pandas as pd


def _full_filename(filename):
    cwd = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    return cwd.joinpath("data", filename)


def leukemia():
    """Load the leukemia dataset.

    These data are taken from Table 1.1 of [1]_.

    The data consist of times of remission (in weeks) of two groups of leukaemia
    patients. Out of the 42 total patients, 21 were in a control group, and the
    other 21 were treated with the drug 6-mercaptopurine. Right-censoring is
    common in the treatment group, but there is no censoring in the control
    group.

    Patients were observed until their leukemia symptoms relapsed or until the
    study ended, whichever occurred first. Each patient in the control group
    experienced relapse before the study ended, while 12 patients in the
    treatment group did not come out of remission during the study. Thus, there
    is heavy right-censoring in the treatment group and no right-censoring in
    the control group.

    Returns
    -------
    pandas.DataFrame
        The leukemia data. Column descriptions:
            * time
                The observed leukemia remission times.
            * status
                Right-censoring indicators (0=censored, 1=event).
            * group
                Group labels (control or treatment).

    References
    ----------
    .. [1] D. R. Cox and D. Oakes. Analysis of Survival Data. Chapman & Hall,
        London (1984), pp. ix+201.
    """
    return pd.read_csv(_full_filename("leukemia.csv"), header=0,
                       dtype=dict(time="int", status="int", group="category")
                       ).rename_axis("patient")


def channing():
    """Load the Channing House dataset.

    This is the ``channing`` dataset in the R package ``boot``. From the package
    description [1]_:

        Channing House is a retirement centre in Palo Alto, California. These
        data were collected between the opening of the house in 1964 until
        July 1, 1975. In that time 97 men and 365 women passed through the
        centre. For each of these, their age on entry and also on leaving or
        death was recorded. A large number of the observations were censored
        mainly due to the resident being alive on July 1, 1975 when the data
        was collected. Over the time of the study 130 women and 46 men died at
        Channing House. Differences between the survival of the sexes, taking
        age into account, was one of the primary concerns of this study.

    These data feature left truncation because residents entered Channing House
    at different ages, and their lifetimes were not observed before entry. This
    is a biased sampling problem since there are no observations on individuals
    who died before potentially entering Channing House.

    Returns
    -------
    pandas.DataFrame
        The Channing House data. Column descriptions:
            * sex
                Sex of each resident (male or female).
            * entry
                The resident's age (in months) on entry to the centre.
            * exit
                The age (in months) of the resident on death, leaving the centre
                or July 1, 1975 whichever event occurred first.
            * time
                The length of time (in months) that the resident spent at
                Channing House (`time=exit-event`).
            * status
                Right-censoring indicator. 1 indicates that the resident died at
                Channing House, 0 indicates that they left the house prior to
                July 1, 1975 or that they were still alive and living in the
                centre at that date.

    References
    ----------
    .. [1] Angelo Canty and Brian Ripley. boot: Bootstrap R (S-Plus) Functions.
        R package version 1.3-20 (2017).
        `CRAN <https://cran.r-project.org/web/packages/boot/index.html>`__.
    """
    return pd.read_csv(_full_filename("channing.csv"), header=0,
                       dtype=dict(sex="category", entry="int", exit="int",
                                  time="int", status="int")
                       ).rename_axis("resident")
