# Microcalorimetry

This package provides a library of data acquisition and analysis tools for RF power calibrations using microcalorimeters. Included in
this package is:

* A Python scripting API.
* A GUI for data acquisition and analysis.

This package is built using Rocky Mountain Ellipse ([RME](https://github.com/usnistgov/rmellipse)), a project to develop tools for digital traceability at NIST.


Please refer to the [docs](https://pages.nist.gov/microcalorimetry-ipages/development/index.html) for detailed information on how to use the package.

## Introduction
Install with pip or preffered package manager.

```
pip install microcalorimetry
```

### CLI

The command line interface can be accessed with the `ucal` command.

```console
ucal --help
```

### GUI

The GUI is launched via the command line interface.

```console
ucal gui
```

### Python API

The Python API provides a programmable interface for performing data acquisition and analysis.

The ``microcalorimetry.measurements`` submodule provides an interface into RF sweep and DC sweep measurement procedures, as well as tools to parse the raw data.


```Python
import microcalorimetry.measurements.dcsweep as dcsweep
import microcalorimetry.measurements.rfsweep as rfsweep
```

Analysis procedures are in the ``microcalorimetry.analysis`` submodule.

```Python
import microcalorimetry.analysis as analysis
```

Mathematical operations compatable with ([RMEMeas](https://pages.nist.gov/rmellipse-ipages/stable/index.html)) objects are stored in the ``microcalorimetry.math`` submodule.
These are the mathematical models used in the analysis procedures.

```Python
import microcalorimetry.math as mcmath
```

Configuration objects for measurements and analysis procedures are provided in a ``microcalorimetry.configs`` submodule.
```Python
import microcalorimetry.configs as configs
```

Definitions of array structures used throughout the package are in the ``microcalorimetry.arrays`` submodule.
```Python
import microcalorimetry.arrays as arrays
```

This package nativley works with [RMEllipse](https://pages.nist.gov/rmellipse-ipages/stable/index.html)  data types and file formats, but includes functions for exporting into
other file formats.
```Python
import microcalorimetry.export as export
```

The GUI supports a plugin system for adding extra function interfaces. Special data types for adding GUI functions are provided in the ``microcalorimetry.gui_dtypes`` submodule.

```Python
import microcalorimetry.gui_dtypes as gui_dtypes
```

## Authors

Contributors names and contact info

Daniel C. Gray
Zenn C. Roberts
Aaron M. Hagerstrom
