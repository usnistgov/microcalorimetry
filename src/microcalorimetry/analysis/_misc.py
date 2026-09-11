import microcalorimetry.configs as configs
import microcalorimetry.math.rfpower as rfpower
import microcalorimetry.arrays as arrays
from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['compression_check']


def compression_check(
    power: configs.RFSweepLike, power_minus_1dB: configs.RFSweepLike, k: int = 2
) -> tuple[RMEMeas[arrays.RFSweep], plt.Figure]:
    """
    Check the compression ratio of of a measurement.

    Requires a parsed rfsweep at desired power, and another
    at the same frequency list backed off on the source by 1 dBm.

    Parameters
    ----------
    power : configs.RFSweepLike
        Power of sensor at desired output.
    power_minus_1dB : configs.RFSweepLike
        Power measured by sensor backed off by 1dB at source from desired
        output.
    k : int, optional
        Expansion factor to plot. The default is 2.

    Returns
    -------
    compression_ratio : RMEMeas[arrays.RFSweep]
        Compression ratio.
    fig : plt.Figure
        Figure object.

    """

    prop = RMEProp(sensitivity=True)

    get_compression = prop.propagate(rfpower.compression_ratio)

    pwr = configs.RFSweepLike(power).load()
    pwr_m1 = configs.RFSweepLike(power_minus_1dB).load()

    ratio = get_compression(pwr, pwr_m1)

    # do uncertanties in linear lande

    fig, ax = plt.subplots(1, 1)
    unc = ratio.stdunc(k=k).cov
    ax.axhline(0.4, color='r', ls='--', label='Acceptable Limit')
    ax.errorbar(
        ratio.nom.frequency, ratio.nom, yerr=unc, ls='', capsize=5, label='Measured'
    )
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Compression Ratio (dB)')
    ax.set_title('Compression Check')
    ax.legend(loc='best')

    return ratio, fig
