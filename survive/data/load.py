"""Functions for loading the packaged datasets."""

import os
import pathlib

import pandas as pd


# Get full filename
def _full_filename(filename):
    cwd = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    return cwd.joinpath("data", filename)


def load_leukemia():
    """Load the leukemia data from Table 1.1 of Cox & Oakes (1984).

    The data consist of times of remission of two groups of leukaemia patients.
    One group was a control, while the other was treated with the drug
    6-mercaptopurine. Right-censoring is common in the treatment group, but
    there is no censoring in the control group.

    Returns
    -------
    data : pandas.DataFrame
        DataFrame of the leukemia data.
        Column descriptions:
            * Time
                The observed leukemia remission times.
            * Event
                Right-censoring indicators (0=censored, 1=event).
            * Group
                Group indicators (0=control, 1=treatment).

    References
    ----------
        * D. R. Cox and D. Oakes. Analysis of Survival Data. Chapman & Hall,
          London (1984), pp. ix+201.
    """
    return pd.read_csv(_full_filename("leukemia.csv"), header=0)


def load_channing():
    """Load the Channing House retirement home data from Hyde (1980).

    This is the `channing` dataset in the R package `boot`. From the package
    description (Canty & Ripley (2017)):

        Channing House is a retirement centre  in Palo Alto, California.
        These data were collected between the opening of the house in 1964 until
        July 1, 1975. In that time 97 men and 365 women passed through the
        centre. For each of these, their age on entry and also on leaving or
        death was recorded. A large number of the observations were censored
        mainly due to the resident being alive on July 1, 1975 when the data
        was collected. Over the time of the study 130 women and 46 men died at
        Channing House. Differences between the survival of the sexes, taking
        age into account, was one of the primary concerns of this study.

    Returns
    -------
    data : pandas.DataFrame
        DataFrame of the Channing House data.
        Column descriptions:
            * Sex
                Sex of each resident ("Male" or "Female").
            * Entry
                The resident's age (in months) on entry to the centre.
            * Exit
                The age (in months) of the resident on death, leaving the centre
                or July 1, 1975 whichever event occurred first.
            * Time
                The length of time (in months) that the resident spent at
                Channing House.
            * Event
                Right-censoring indicator. 1 indicates that the resident died at
                Channing House, 0 indicates that they left the house prior to
                July 1, 1975 or that they were still alive and living in the
                centre at that date.

    References
    ----------
        * J. Hyde. Testing survival with incomplete observations. Biostatistics
          Casebook. R.G. Miller, B. Efron, B.W. Brown, and L.E. Moses (editors).
          Wiley (1980), pp. 31--46.
        * Angelo Canty and Brian Ripley. boot: Bootstrap R (S-Plus) Functions.
          R package version 1.3-20 (2017).
          CRAN: https://cran.r-project.org/web/packages/boot/index.html
    """
    return pd.read_csv(_full_filename("channing.csv"), header=0)
