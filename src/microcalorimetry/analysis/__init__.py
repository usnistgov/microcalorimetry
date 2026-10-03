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
from ._misc import *

