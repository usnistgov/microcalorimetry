# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:05:27 2026

@author: dcg2
"""


def plot_RMEMeas(file, hdf5_path, fig=None):
    """
    Generic plot function for RMEMeas objects stored in HDF5.

    Parameters
    ----------
    hdf5_path : TYPE
        DESCRIPTION.
    fig : matplotlib figure
        Figure to plot to. If None, a new figure will be created.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    figs = []
    plotted = False

    with h5py.File(file, 'r') as f:
        data = f[hdf5_path]

        data = RMEMeas.from_h5(f[hdf5_path])
        if len(data.nom.shape) == 1 and data.nom.dtype is not complex:
            print('1d array, plotting as line')
            if not fig:
                fig, ax = plt.subplots(1, 1)
                figs.append(fig)
            else:
                ax = fig.axes[0]

            stdunc = data.stdunc().cov
            xlabel = data.nom.dims[0]
            ylabel = hdf5_path.split('/')[-1]
            xvals = data.nom.coords[xlabel]
            ax.errorbar(
                xvals,
                data.nom,
                yerr=stdunc,
                fmt='o',
                capsize=3,
                label='(k=1) .../' + '/'.join(hdf5_path.split('/')[-2:]),
            )
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(hdf5_path)
            ax.legend(loc='best')
            fig.tight_layout()

        elif (
            len(data.nom.shape) == 2
            and data.nom.dtype is not complex
            and data.nom.shape[1] == 1
        ):
            data = data[:, 0]
            if not fig:
                fig, ax = plt.subplots(1, 1)
                figs.append(fig)
            else:
                ax = fig.axes[0]
            stdunc = data.stdunc().cov[:,1]
            xlabel = data.nom.dims[0]
            ylabel = hdf5_path.split('/')[-1]
            xvals = data.nom.coords[xlabel]
            ax.errorbar(
                xvals,
                data.nom,
                yerr=stdunc,
                fmt='o',
                capsize=3,
                label='(k=1) .../' + '/'.join(hdf5_path.split('/')[-2:]),
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(hdf5_path)
            ax.legend(loc='best')
            fig.tight_layout()
        elif len(data.nom.shape) == 2 and data.nom.dtype is not complex:
            data_full = data
            if not fig:
                fig, axs = plt.subplots(1, 2)
                figs.append(fig)
            else:
                axs = fig.axes
            for i, ax in enumerate(axs):
                data = data_full[:, i]
                ub = data.uncbounds(k=1)[0]
                lb = data.uncbounds(k=-1)[0]
                xlabel = data_full.nom.dims[0]
                ydim = data_full.nom.dims[1]
                ylabel = ydim + ' : ' + str(data_full.nom.coords[ydim][i].values)
                xvals = data.nom.coords[xlabel]
                ax.plot(
                    xvals,
                    data.nom,
                    'o-',
                    lw=2,
                    label='.../' + '/'.join(hdf5_path.split('/')[-2:]),
                )
                ax.plot(xvals, lb, '--k', label='k = 1')
                ax.plot(xvals, ub, '--k')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.legend(loc='best')
            fig.suptitle(hdf5_path)
            fig.tight_layout()
        else:
            print('Couldnt plot shape/dtype')

    return tuple(figs)