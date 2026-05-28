import matplotlib.pyplot as _plt
import matplotlib as _mpl

_SMALL_SIZE = 8
_MEDIUM_SIZE = 10
_BIGGER_SIZE = 12
_LINE_WIDTH = 3
_X_TIKS = 6
_Y_TIKS = 5

# this sets the parameters
_mpl.rcParams['figure.figsize'] = (8,8)
_plt.rc('font', size=_SMALL_SIZE)          # controls default text sizes
_plt.rc('axes', titlesize=_MEDIUM_SIZE)     # fontsize of the axes title
_plt.rc('axes', labelsize=_MEDIUM_SIZE)    # fontsize of the x and y labels
_plt.rc('xtick', labelsize=_MEDIUM_SIZE)    # fontsize of the tick labels
_plt.rc('ytick', labelsize=_MEDIUM_SIZE)    # fontsize of the tick labels
_plt.rc('legend', fontsize=_SMALL_SIZE)    # legend fontsize
_plt.rc('figure', titlesize=_BIGGER_SIZE)  # fontsize of the figure title