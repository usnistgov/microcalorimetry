RF Levelling and Statistics
===========================
The levelling and statistics settings for an RF sweep determine how long each step goes,
when the step is considered stable, and configure the feedback loop for power levelling.


Statistics
----------

The statistics configuration tells the runner how to determine when a step is stable, minimum wait times,
and statistics settings. 

If the **use_traditional_stats** field is True, then the runner uses a statistical test to determine stability. Otherwise
each step waits for the minimum amount of time then moves on to the next frequency point.

.. csv-table:: Statistics Example
   :file: stats.csv
   :header-rows: 1

Levelling
---------

The levelling configuration tells the runner what power signal to try and level to (DUT_power, monitor_power, etc), as well as configures
the feedback loop. It also lets you set hard limits across all frequency points based on every power signal in the calorimeter.

.. csv-table:: Levelling Example
   :file: levelling.csv
   :header-rows: 1
