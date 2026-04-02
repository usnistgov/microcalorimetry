Package Structure
=================

The package is seperated into a high level and low level API.

The high level API is the primary interface of functions that interact
with user interfaces (UX) - i.e the GUI or the CLI. The high level api is also available for use via
importing the package and invoking from a python script.

In general, the job of functions in the high level is to take in configuration objects
and basic python data types suitable for a user interface (strings, numbers,
booleans) and perform some set of operations based on
those configurations. Things like "perform an RF sweep measurement" or
"generate an effective efficiency data file." These functions will take
in configuration objects and coerce input data into acceptable in memory
representations needed for the lower level API.

The low level API contains a set of mathematical and generic utilty functions
and classes. These functions and classes are handed off to by the higherlevel API,
and can't be directly interacted with by a UX. However, they are available for use in a python script by import.

Configuration objects act like adapters between the high-level API and the low
level API. High level functions take in configuration objects, which they pass off to lower level API.
In the case of measurement running functions, the lower level
API include calls to instrument control drivers and interfaces that manage
the flow of an experiment. In the case of high level functions that process
data, the high level functions pass of data to mathematical functions
according to the appropriate algorithm.

Saving data is controlled at the UX level, and is done by specifying output
paths. When writing a python script, you must manually save output data into the
desired format.

.. raw:: html
    :file: processing_flow.drawio.svg


The goal of this architecture is to facilitate adding new analysis functions
that operate will with both the UX and via Python scripts.This hopes to make the code
modular. You can interact with the high level API and UX for routine tasks, and
interact at a lower level for more niche analysis.


Submodules that contain the high level API functions:

* microcalorimetry.analysis
* microcalorimetry.measurements
* microcalorimetry.export

Submodules that contain lower level mathematical functions:

* microcalorimetry.math

Submodules that contain configuration objects:

* microcalorimetry.configs

