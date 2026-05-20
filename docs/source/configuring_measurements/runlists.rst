Measurement Run Lists
=====================

Both dc sweep and RF sweep measurements require run lists to be defined.

For dc sweeps, these define the source settings in amps. For RF sweeps, these
define the frequency and power levels.


DC Sweep Run List
-----------------

For dc sweep runlists, specify a single column CSV where each row represents
the source current value across the resistor connected to the SMU in the heater role.

DC Sweeps should be book ended by 0 source values to get a good off measurement. For example,

.. csv-table:: Sample DC Sweep Run List
   :file: dc_runlist.csv
   :header-rows: 1


RF Sweep Run List
-----------------

For rf sweep run lists you need to specify the frequency, initial starting power of the source,
the target power for levelling (if it's turned on) and the source limit for that frequency point.

If a row contains all zeroes, it is considered off and the RF Source is fully shut down for that
row. The set of measurements between off measurements are called a referred to as a segment. Segments
should be book ended by at least 2 off rows to get a good off measurement of the calorimeter's thermopile.

.. csv-table:: Sample RF Sweep Run List
   :file: rf_runlist.csv
   :header-rows: 1