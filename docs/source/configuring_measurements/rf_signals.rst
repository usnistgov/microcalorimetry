RF Sweep Signals
================

In an RF sweep measurement there is a requirement to estimate the current power of the different active sensors to support he feedback loop
for power levelling, and to shut down in the event of unsafe power levels. To do this, the configuration for an RF sweep measurement requires you
to define power signals. Power signals associate a type of power sensor with their associated instruments, data columns, and additional data required
to estimate RF power like thermoelectric fit coefficients for thermo electric sensors, bias resistance for bolometers, or the AM modulation sensitivity
for RF sources.

These signal configurations are used by both the measurement runner and the parser to identify which instruments and data columns are associated with
what type of sensors.

Bolometers require a dc bias resistance to be specified, along with a bias_monitor instrument and its associated voltage column. This example
shows the specification of a bolometer sensor as the DUT operating at 200 Ohms. A bolometer power signal requires input from a **vdc** (dc voltage) signal,
voltage signal, which has it's instrument,units, and data column specified.

.. csv-table:: Bolometer Signal
   :file: DUT_signal_bolometer.csv
   :header-rows: 1

Thermoelectric sensors in a calorimeter or sensor require a voltage signal to process data. For thermoelectric signal, the coefficients are provided
as an estimate of the linear sensitivity in units of :math:`\frac{\mathrm{V}}{\mathrm{W}}` which is sufficient for the purposes of power levelling.

.. csv-table:: Thermoelectric Signal
   :file: clrm_signal_thermoelectric.csv
   :header-rows: 1

Some sensors may require multiple input quantities. For example, temperature dependent thermo electric sensors require a the voltage and current
from a thermometer as an input, as well as tg voltage from a thermoelectric. Those can be specified as well.

.. csv-table:: Temperature Dependent Thermoelectric Signal
   :file: dut_signal_thermoelectric_temperature.csv
   :header-rows: 1

RF Sources are commanded to output a power, but also may have there output power modulated by a Voltage Source acting as an **rf_amplitude_adjuster**.

.. csv-table:: RF Source with AM Signal
   :file: rf_source_signal.csv
   :header-rows: 1


