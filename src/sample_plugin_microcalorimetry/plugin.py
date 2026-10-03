from microcalorimetry.gui_dtypes import Folder, SaveAsPath
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def primitive_types(
    period: float,
    amplitude: float = 1,
    title: str = 'Sin Wave Example',
    line_width: float = 3,
    color: str = 'k',
) -> tuple[plt.Figure, np.array]:
    """
    Use primitive types to make a sin wave.

    More descriptions here.

    Parameters
    ----------
    period : float
        _description_
    amplitude : float, optional
        _description_, by default 1
    title : str, optional
        _description_, by default "Sin Wave Example"
    line_width : float, optional
        _description_, by default 3
    color : str, optional
        _description_, by default 'k'

    Returns
    -------
    figure : plt.Figure
        Figure generated.
    y : np.array
        1 dimensional array of y-values
    """
    if period is None:
        period = np.random.random()
    x = np.linspace(0, np.pi * 2, 100)
    y = amplitude * np.sin(x / period)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, color=color, lw=line_width)
    ax.set_title(title)
    print('Exiting function')
    return fig, y


# dictionary of names and functions to
# add to a tab
FUNCTIONS = {'primitive_types': primitive_types}

# dictionary of functions that should be outputting group saveable stuff
OUTPUT_GROUP_SAVEABLE = ['primitive_types']
