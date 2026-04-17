"""
The measurements submodule contains functions for interacting with specific measurements.

Each submodule within contains, at minimum, three functions for a particular measurment.

* ``run``: Takes in a configuration file and performs a measurement
* view: Takes in the output of an ongoing or complete measurement and generates plots to explore the measurement
* parse: Takes in raw data and outputs data to a well defined intermediary data-set with uncertainties for further processing.

Additional methods may be provided depending on the measurement.
"""
from ._common import *