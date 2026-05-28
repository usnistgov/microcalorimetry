Instruments
===========

Measurement configurations utilize ExperimentParameters, which are a way of describing dictionary structures
using CSV files defined in the `rminstr <https://pages.nist.gov/rminstr-ipages>`_  package. 

All the configurations for every instrument, along with any other settings, are provided as a list of files to the measurement runner. These
are concatenated into a single large dictionary that is used by the runner to run the measurement. These files include things like:

* What instruments need to be connected to?
* What are the instruments settings at different points in the measurement?
* What is the important metadata about the measurement?
* What are the instruments doing (i.e what is their role) in the measurement?

The required fields for a given measurement are defined using JSON schema, which are given in the configuration reference.

They follow the general structure of

* A set of general configurations corresponding to the overarching measurement.
    * Data column, required metadata, etc.
* A set of instrument configurations that specify every controlled instrument in the measurement.
    * Models, serials, roles, etc.

Instruments can have a type and a role. For example, a Voltmeter is a type of instrument, but doesn't describe what it's doing in the measurement.
The role describes what it's doing, which could be a **thermopile_monitor** or a **bias_monitor**.

The type of instrument is directly related to the measurement functionality interfaces defined in `rminstr <https://pages.nist.gov/rminstr-ipages>`_ .
Settings a given instrument can utilize are directly defined by the settings available for different measurement interfaces defined by that package.

Every instrument needs an **initial_settings** field to define what the instrument should be initialized too. Measurement runners may require additional operational mode fields to define what the instrument
settings should be for different modes in the measurement. For example, an RF sweep has a **monitor_mode** during its initial warm up at every frequency
point while the thermopile comes to equilibrium and a **fast_off** mode that occurs at the end of every frequency point.

For example, the HP HP3458A could be used to as a **thermopile_monitor** to monitor the voltage of a thermoelectric,
or as a **bias_monitor** to monitor the voltage of a Type IV power meter. Below is an example of an instrument configuration that
configures the an HP 3458A to operate as a **bias_monitor**.

.. csv-table:: HP3458A BiasMonitor
   :file: HP3458A_BiasMonitor.csv
   :header-rows: 1


