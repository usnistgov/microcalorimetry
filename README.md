# Microcalorimetry

> [!NOTE]
> This software is in active and early development by the RF power calibrations service at NIST to support
> RF power calibrations and the development primary RF power standards. Expect breaking breaking changes as the software
> evolves. Instrument interfaces are added and tested as needed for the calibration service. Bugs may
> be present in the instrument interfaces that we are unaware of. Please exercise caution when using interfaces
> presented in this code.

This package provides a library of data acquisition and analysis tools for RF power calibrations using microcalorimeters. Included in
this package is:

* A Python scripting API
* A User Inteface (CLI and GUI) for data acquisition and analysis.

This package is built using Rocky Mountain Ellipse ([RME](https://github.com/usnistgov/rmellipse)), a project to develop tools for digital traceability at NIST.


Please refer to the [docs](https://pages.nist.gov/microcalorimetry-ipages/development/index.html) for detailed information on how to use the package.

## Introduction
Install with pip or preffered package manager.

```
pip install microcalorimetry
```

### CLI

The command line interface is can be accessed with the `ucal` command:

```console
ucal --help
```

### GUI

The GUI is launched via the command line interface

```console
ucal gui
```

### Python API

The Python API provides a functional interface for performing measurements and data analysis.

The ``microcalorimetry.measurements`` submodule provides an interface into RF sweep and DC sweep measurement procedures, as well as tools to parse the raw data.


```Python
import microcalorimetry.measurements.dcsweep as dcsweep
import microcalorimetry.measurements.rfsweep as rfsweep
```

Analysis functions that take in parsed data and generate new data sets with uncertainties (like the effective efficiency of  power sensors) are provided in the ``microcalorimetry.analysis`` submodule.

```Python
import microcalorimetry.analysis as analysis
```


Configuration objects for measurements and analysis scripts are provided in a ``microcalorimetry.configs`` module.
```Python
import microcalorimetry.configs as configs
```

Mathematical operations compatable with ([RMEMeas](https://pages.nist.gov/rmellipse-ipages/stable/index.html)) objects are stored in the ``microcalorimetry.math`` submodule.

```Python
import microcalorimetry.math as mcmath
```

## Authors

Contributors names and contact info

Daniel C. Gray, Zenn C. Roberts, Aaron M. Hagerstrom
