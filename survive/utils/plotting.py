"""Plotting helper functions."""


def add_legend(ax, legend_kwargs=None):
    """Add a legend to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to add the legend.
    legend_kwargs : dict, optional
        Dictionary of keyword parameters to pass to
        :meth:`matplotlib.axes.Axes.legend`.

    Returns
    -------
    legend : matplotlib.legend.Legend
        The :class:`matplotlib.legend.Legend` instance added to the plot.
    """
    legend_params = dict(loc="best", frameon=True, shadow=True)
    if legend_kwargs is not None:
        legend_params.update(legend_kwargs)
    return ax.legend(**legend_params)
