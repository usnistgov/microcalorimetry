"""
This module contains high-level routines for performing microcalorimetry analysis.

This module should contain static functions which take in configuration objects,
and primative types (strings, numbers) that are then used to process the data.
Where possible, they should use math functions defined in the math module.

These functions should be interfaceable via either a command line interface or
a GUI where the user can only enter string-like inputs.

These function should utilize DataModelContainers when referring to inputs that represent inputs that cant be described
by a short string (i.e. something convenient for a command line interface). This
enables the passing of in-memory representations
of data via the DataModelContainers so that it isn't required to serialize the output of one function
in order to pass it to the input of another.

"""

from ._correction_factors import *
from ._sensitivity import *
from ._calc_eta import *

import matplotlib.pyplot as _plt
import matplotlib as _mpl

_SMALL_SIZE = 8
_MEDIUM_SIZE = 10
_BIGGER_SIZE = 12
_LINE_WIDTH = 3
_X_TIKS = 6
_Y_TIKS = 5

# this sets the parameters
_mpl.rcParams['figure.figsize'] = (8,8)
_plt.rc('font', size=_SMALL_SIZE)          # controls default text sizes
_plt.rc('axes', titlesize=_MEDIUM_SIZE)     # fontsize of the axes title
_plt.rc('axes', labelsize=_MEDIUM_SIZE)    # fontsize of the x and y labels
_plt.rc('xtick', labelsize=_MEDIUM_SIZE)    # fontsize of the tick labels
_plt.rc('ytick', labelsize=_MEDIUM_SIZE)    # fontsize of the tick labels
_plt.rc('legend', fontsize=_SMALL_SIZE)    # legend fontsize
_plt.rc('figure', titlesize=_BIGGER_SIZE)  # fontsize of the figure title