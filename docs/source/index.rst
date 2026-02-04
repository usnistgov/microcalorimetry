.. Rocky Mountain Ellipse documentation master file, created by
   sphinx-quickstart on Wed Jul 24 09:45:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Microcalorimetry
================

This package provides a library of data acquisition and analysis tools for RF power calibrations using microcalorimeters. Included in
this package is:

* A Python scripting API
* A User Inteface (CLI and GUI) for data acquisition and analysis.

This package is built using Rocky Mountain Ellipse  (`RME <https://github.com/usnistgov/rmellipse>`_)
, a project to develop tools for digital traceability at NIST.

Introduction
------------
Install with pip or preffered package manager.

.. code-block:: console

   pip install microcalorimetry


The command line interface is can be accessed with the `ucal` command:

.. code-block:: console

   ucal --help



The GUI is launched via the command line interface

.. code-block:: console

   ucal gui



The Python API provides a functional interface for performing measurements and data analysis.

The ``microcalorimetry.measurements`` submodule provides an interface into RF sweep and DC sweep measurement procedures, as well as tools to parse the raw data.


.. code-block:: python

   import microcalorimetry.measurements.dcsweep as dcsweep
   import microcalorimetry.measurements.rfsweep as rfsweep


Analysis functions that take in parsed data and generate new data sets with uncertainties (like the effective efficiency of  power sensors) are provided in the ``microcalorimetry.analysis`` submodule.

.. code-block:: python

   import microcalorimetry.analysis as analysis



Configuration objects for measurements and analysis scripts are provided in a ``microcalorimetry.configs`` module.

.. code-block:: python

   import microcalorimetery.configs as configs


Mathematical operations compatable with
`RMEMeas <https://pages.nist.gov/rmellipse-ipages/stable/index.html>`_
objects are stored in the ``microcalorimetry.math`` submodule.

.. code-block:: python

   import microcalorimetery.math as mcmath


Table of Contents
=================

.. toctree::
   :maxdepth: 3

   package_structure/index.rst
   schemas/index.rst
   auto_examples/index.rst
   cli.rst


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

