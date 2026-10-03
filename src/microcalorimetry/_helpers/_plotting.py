# -*- coding: utf-8 -*-
"""
Helper functions for plotting
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def colored_errbar(
        ax: plt.Axes,
        *,
        x: np.array,
        y: np.array,
        yerr: np.array,
        c: np.array,
        zorder: int,
        line_width: float = 1.5,
        capsize: float = 3,
        cmap: str = 'viridis',
        errbar_kwargs: dict = None,
        scatter_kwargs: dict = None
        )->tuple:
    """
    Scatter plot with color values and a color mapping and matching color errorbars.

    Parameters
    ----------
    x : np.array
        DESCRIPTION.
    y : np.array
        DESCRIPTION.
    c : np.array
        DESCRIPTION.
    ax : plt.Axes
        DESCRIPTION.
    zorder : int
        DESCRIPTION.
    cmap : str, optional
        DESCRIPTION. The default is 'viridis'.
    errbar_kwargs : dict, optional
        DESCRIPTION. The default is None.
    scatter_kwargs : dict, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    tuple:
        Tuple of plot artist handles (scatter, caps, bars)
    """
    if errbar_kwargs is None:
        errbar_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=c.min(), vmax=c.max())
    colors = cmap(norm(c))           # Array of RGBA colors

    
    # plot the scatter points
    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, **scatter_kwargs)
    
    #plot the error bars structure (set color to 'none' so they are invisible initially)
    # Note: we catch the container elements to manipulate them
    dot, caps, bars = ax.errorbar(x, y, 
                                  yerr=yerr, fmt='none', ecolor='none', 
                                  elinewidth = line_width,
                                  capsize = capsize, 
                                  zorder = zorder -1,
                                  **errbar_kwargs)
    
    # extract error bar line segments and recolor them via LineCollection
    # bars[0] contains the vertical error bar LineCollection
    if line_width:
        segments = bars[0].get_segments()
        colored_bars = LineCollection(segments, colors=colors, linewidths=line_width)
        ax.add_collection(colored_bars)
    
    # (Optional) Recolor the caps to match the points
    # The caps list contains two elements per point (top and bottom caps)
    if capsize:
        for i, color in enumerate(colors):
            caps[2*i].set_color(color)
            caps[2*i+1].set_color(color)

    return sc, caps, bars