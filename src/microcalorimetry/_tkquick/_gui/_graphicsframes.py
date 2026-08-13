# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:23:39 2024

@author: dcg2
"""

from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import h5py
import os.path
from microcalorimetry._tkquick._gui._themes import console_font
import microcalorimetry.math.vna as vna
import microcalorimetry.math.trig as trig
import microcalorimetry.math.rmemeas_extras as rmemeas_extras
from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
from rmellipse.utils import load_object
import microcalorimetry._gwex as gwex
import xarray as xr

customtkinter = ctk
# Implement the default Matplotlib key bindings.
# from stagefieldframes import AnalysisStageFieldFrame

# plt.ioff()

PLACEHOLDER_TEXT = 'Plots'
CLOSE_PLOT = 'x'
PERSISTENT_TABS = [PLACEHOLDER_TEXT, CLOSE_PLOT]


class PlotsFrame(customtkinter.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        # frame to store little menu drop downs
        self.top_frame = ctk.CTkFrame(self, fg_color='transparent')
        self.top_frame.grid(row=0, column=0, sticky='nesw', pady=10)

        self.close_menu = ctk.CTkOptionMenu(
            self.top_frame,
            values=['active', 'all'],
            command=self.close_callback,
        )
        self.close_menu.grid(row=0, column=0, padx=(0, 10), sticky='nwe', pady=10)
        self.close_menu.set('close')

        # export plots
        self.export_menu = ctk.CTkOptionMenu(
            self.top_frame,
            values=['snapshot all', 'pickle all', 'pickle', 'unpickle'],
            command=self.export_callback,
        )
        self.export_menu.grid(row=0, column=1, padx=(0, 10), sticky='nwe', pady=10)
        self.export_menu.set('file')

        # plot selector, spans the whole row

        self.sel_frame = ctk.CTkFrame(self, fg_color='transparent')
        self.sel_frame.grid(row=1, column=0, sticky='nesw', pady=10)
        self.sel_frame.columnconfigure(0, weight=1)

        self.selector = ctk.CTkOptionMenu(
            self.sel_frame, values=[], command=self.selector_callback
        )
        self.selector.grid(row=1, column=0, sticky='nesw', columnspan=100)
        self.selector.set('')

        # shows the active tabs
        # this should only be communicated with through
        # this frame, as it the tool bar widgets need to be updated
        # accordingly
        self._tabview: GraphicsTabs = GraphicsTabs(self)
        self._tabview.grid(row=2, column=0, sticky='nesw')

    def selector_callback(self, choice):
        self._tabview.set(choice)

    def add_plot_to_selector(self, name):
        old_values = self.selector.cget('values')
        values = [name] + old_values
        self.selector.configure(values=values)
        self.selector.set(name)

    def add_hidden_tabs(self):
        open_labels = plt.get_figlabels()
        for ol in open_labels:
            if ol not in self._tabview.plots_dict:
                print('Caught hidden tab, ', ol, ',  adding i it')
                fig = plt.figure(ol)
                name = str(plt.figure(ol).number)
                self.add_plot(fig, 'FAILED' + name)

    def add_dummy_plot(self, name):
        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0, 3, 0.01)
        fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
        self.add_plot(fig, name)

    def remove_plot_from_selector(self, name):
        old_values = self.selector.cget('values')
        old_values.remove(name)
        self.selector.configure(values=old_values)

    def clear_selector(self):
        self.selector.configure(values=[])
        self.selector.set('')

    def add_plot(self, fig: plt.Figure, name: str):
        actual_name = self._tabview.add_plot(fig, name)
        self.add_plot_to_selector(actual_name)

    def export_callback(self, choice):
        self.export_menu.set('file')
        match choice:
            case 'snapshot all':
                self.plotmenu_export_all()
            case 'pickle all':
                self.pickle_all_figs()
            case 'unpickle':
                self.unpickle_figs()
            case _:
                raise Exception(f'{choice} not recognized')

    def close_callback(self, choice):
        self.close_menu.set('close')
        match choice:
            case 'all':
                self._tabview.close_all_tabs()
                self.clear_selector()
            case 'active':
                self.remove_plot_from_selector(self._tabview.get())
                self._tabview.close_open_tab()
            case _:
                raise Exception(f'{choice} not recognized')

    def plotmenu_export_all(self):
        dialog = ctk.CTkInputDialog(
            text='enter a format [.pdf, .png]', title='Export Format'
        )
        text = dialog.get_input()
        folder = str(
            ctk.filedialog.askdirectory(
                initialdir=str(Path.cwd()),
                title='Export plots to folder',
            )
        )

        if folder != '':
            folder = Path(folder)
            frmt = text
            plots_dict = self._tabview.plots_dict
            for name, fig in plots_dict.items():
                fig.savefig(folder / f'{name}{frmt}')
        else:
            print('no folder selected.')

    def pickle_all_figs(self):
        """Pickles all the open folders to a figure."""
        import pickle

        folder = str(
            ctk.filedialog.askdirectory(
                initialdir=Path.cwd(),
                title='Select folder to save pickle figures to.',
            )
        )

        if folder != '':
            folder = Path(folder)
            plots_dict = self._tabview.plots_dict
            for name, fig in plots_dict.items():
                with open(folder / f'{name}.pklfig', 'wb') as f:
                    pickle.dump(fig, f)
        else:
            print('no folder selected.')

    def unpickle_figs(self):
        """Pickles all the open folders to a figure."""
        import pickle

        files = ctk.filedialog.askopenfilenames(
            title='(Only select files you made, pickling can be unsafe) Select pickle objects to open:',
            filetypes=[('Pickled Figure', '.pklfig')],
            initialdir=str(Path.cwd()),
        )

        for fn in files:
            with open(fn, 'rb') as f:
                fig_loaded = pickle.load(f)
            self.add_plot(fig_loaded, Path(fn).stem)


class GraphicsTabs(customtkinter.CTkTabview):
    """Container where plots are viewed."""

    def __init__(self, master, **kwargs):
        super().__init__(master, command=self.close_open_tab, **kwargs)

        # # datafile viewer tab
        # self.add("HDF5")
        # self.tab("HDF5").grid_rowconfigure(1, weight=1)
        # self.tab("HDF5").grid_columnconfigure(0, weight=1)
        # self.hdf5viewer = HDF5viewer(master=self.tab('HDF5'))

        # other stuff
        self.last_opened_tab = None
        self.all_tabs = []  # , 'HDF5']
        self.plots_dict = {}
        self.parent = master

    def get_tab_by_ind(self, ind):
        for tab in self.all_tabs:
            if self.index(tab) == ind:
                return tab
        raise ValueError('Index not in tabs')

    def close_all_tabs(self):
        tabnames = list(self.plots_dict.keys())
        for tabname in tabnames:
            try:
                self.delete(tabname)
            except Exception as e:
                print(f'Failed to close {tabname} for {e}')
        self.close_hidden_tabs()

    def close_open_tab(self, name: str = None):
        if name is None:
            name = self.get()

        index = self.index(name)
        # change open tab before deleting
        # if we are deleting the open tab
        if index > 0 and name == self.get():
            new_tab_ind = index - 1
            new_tab = self.get_tab_by_ind(index - 1)
            self.set(new_tab)
        self.delete(name)
        self.close_hidden_tabs()
        print(len(plt.get_fignums()), ' open figures.')

    def close_hidden_tabs(self):
        open_labels = plt.get_figlabels()
        for ol in open_labels:
            if ol not in self.plots_dict:
                print('Caught hidden tab, ', ol, ', closing it')
                plt.close(ol)

    def tab_exists(self, name):
        try:
            self.tab(name)
            return True
        except Exception:
            return False

    def delete(self, name):
        # delte from tab view
        super().delete(name)
        # close out the figure in the backend
        try:
            plt.close(self.plots_dict[name])
        except KeyError:
            pass
        # remove from any dictionairys/lists  that are tracking things
        self.all_tabs.remove(name)
        self.plots_dict.pop(name)

    def add_plot(self, fig, name):
        count = 0
        new_name = name
        while self.tab_exists(new_name):
            new_name = name + str(count)
            count += 1
        name = new_name
        fig.set_label(name)
        self.add(name)
        self.set(name)
        self.plots_dict[name] = fig
        self.tab(name).grid_rowconfigure(0, weight=1)
        self.tab(name).grid_columnconfigure(0, weight=1)
        self.all_tabs.append(name)
        root = self.tab(name)

        canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # a custom plot tool bar definition may have been passed in
        from microcalorimetry._tkquick._api import _APP

        custom_toolbar = _APP.plots_toolbar
        if custom_toolbar is None:
            toolbar = NavigationToolbar2Tk(canvas, root)
        else:
            toolbar = custom_toolbar(canvas, root, self.tab(name))

        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        def on_key_press(event):
            print('you pressed {}'.format(event.key))
            key_press_handler(event, canvas, toolbar)

        canvas.mpl_connect('key_press_event', on_key_press)
        # hide tabs
        self._top_spacing = 0
        self._top_button_overhang = 0
        self._segmented_button.grid_forget()
        # self._configure_grid()

        def _quit():
            root.quit()  # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent
            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        return name
        # print(len(plt.get_fignums()), " open figures.")


# class ConsoleRedirector:
#     def __init__(self, widget):
#         self.widget = widget

#     def write(self, text):
#         self.widget.insert(tk.END, text)
#         self.widget.see(tk.END)  # Auto-scroll to the bottom


class PlaceHolderFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)


class HDF5GroupRow:
    def __init__(self, position, group, master, depth, hdf5_file, **kwargs):
        self.hdf5_file = hdf5_file

        # make a call button that goes down 1 level
        is_RMEMeas = False
        dfm = '-'

        if '__class__.__name__' in group.attrs.keys():
            cname = group.attrs['__class__.__name__']
            is_RMEMeas = cname == 'RMEMeas' or cname == 'MUFmeas'

        self.name = group.name
        self.position = position
        self.group = group
        self.master = master
        # make lil nametas
        if not is_RMEMeas:
            self.objbutt = ctk.CTkButton(
                master=master,
                text=' ' * 7 + group.name.split('/')[-1],
                fg_color='transparent',
                anchor='w',
                command=self.go_down,
            )
            self.plot = ctk.CTkLabel(master=master, text='')
            self.plot.grid(row=position, column=2)

        else:
            self.objbutt = ctk.CTkLabel(
                master=master,
                text=' ' * 7 + group.name.split('/')[-1],
                justify='left',
                anchor='w',
            )
            self.plot = ctk.CTkSegmentedButton(
                master=master,
                values=['+', ']', 'u'],
                command=self.make_plot,
                width=20,
                height=20,
            )
            self.plot.grid(row=position, column=2)
            print(self.plot.get())

        self.objbutt.grid(row=position, column=0, columnspan=2, sticky='ew')

        # button to delete groups
        self.edit_but = ctk.CTkOptionMenu(
            master=master,
            width=50,
            height=20,
            values=['delete', 'clip'],
            command=self.edit,
        )
        self.edit_but.set('edit')
        self.edit_but.grid(row=position, column=1, sticky='e')

        # make a metadata page
        color = self.master.cget('fg_color')[0]
        self.metabut = ctk.CTkButton(
            master=master,
            text='',
            command=self.make_meta,
            width=20,
            height=20,
            fg_color=color,
        )
        self.metabut.grid(row=position, column=3)

        # print('HDF5 row ', position, ' for ', group.name, is_RMEMeas)

    def edit(self, choice):
        reset = True
        match choice:
            case 'delete':
                print('deleting : ', self.name)
                with h5py.File(self.hdf5_file, 'a') as f:
                    del f[self.name]
                self.master.refresh()
                reset = False
            case 'clip':
                import subprocess

                path = str(Path(self.hdf5_file)) + self.name
                print('Copying', path, 'to clip')
                subprocess.run('clip', input=path, check=True, encoding='utf-8')

            case _:
                print(choice, 'not defined')
                return
        if reset:
            self.edit_but.set('edit')

    def destroy(self):
        for item in [self.objbutt, self.plot, self.edit_but, self.metabut]:
            item.destroy()

    def go_down(self):
        # print('callback from ', self.name, ' row ', self.position)
        self.master.build(path=self.name)

    def make_plot(self, value):
        from microcalorimetry._tkquick._api import _APP

        # make a new plot
        self.plot.set(None)
        plot_name = f'{Path(self.hdf5_file).stem}{".".join(self.name.split("/"))}'
        if value == '+':
            output = plot_RMEMeas(self.hdf5_file, self.name)
            for item in output:
                if isinstance(item, plt.Figure):
                    _APP.graphicsframe.add_plot(item, plot_name)
        # try to plot RMEMeas object onto the active figure
        elif value == ']':
            open_tab = self.master.parent.graphicstabs.get()
            if open_tab == 'x' or open_tab == PLACEHOLDER_TEXT:
                raise Exception('Cant plot onto open tab')
            figure = self.master.parent.graphicstabs.plots_dict[open_tab]
            output = plot_RMEMeas(self.hdf5_file, self.name, fig=figure)
            figure.canvas.draw()
            figure.canvas.flush_events()
        elif value == 'u':
            output = uncertainty_breakdown(self.hdf5_file, self.name)
            for item in output:
                if isinstance(item, plt.Figure):
                    _APP.graphicsframe.add_plot(item, plot_name)

    def make_meta(self):
        with h5py.File(self.master.hdf5_file, 'r') as f:
            grp = f[self.name]
            count = 0
            for a in grp.attrs:
                print('\n' + a + '\n' + len(a) * '---')
                print(grp.attrs[a])
                count += 1
            if not count:
                print('No meta data for group: ', self.name)
            plots_dict = {}


class HDF5viewer(customtkinter.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.parent = master
        self.hdf5_file = Path.home()
        # add widgets onto the frame...
        # self.label = customtkinter.CTkLabel(self,text = '>> TERMINAL OUTPUT')
        # self.label.grid(row=0, column=0, padx=20)
        # create scrollable textbox
        self.grid_columnconfigure(0, weight=2)
        # self._scrollbar.configure(width = 0)
        # refresh button
        self.root_button = ctk.CTkButton(
            master=self, command=self.build, text='root', fg_color='transparent'
        )
        self.root_button.grid(row=0, column=0, columnspan=1, sticky='nesw')

        self.refresh_button = ctk.CTkButton(
            master=self, command=self.refresh, text='refresh', fg_color='transparent'
        )
        self.refresh_button.grid(row=0, column=1, columnspan=1, sticky='nesw')

        self.file_button = ctk.CTkButton(
            master=self, command=self.set_file, text='open_file', fg_color='transparent'
        )
        self.file_button.grid(row=0, column=2, columnspan=2, sticky='nesw')

        # make the column headers
        self.label = ctk.CTkLabel(master=self, text='Objects', justify='left')
        self.label.grid(row=1, column=0, padx=(0, 10), pady=10, sticky='ew')
        self.label = ctk.CTkLabel(master=self, text='Del', justify='left')
        self.label.grid(row=1, column=1, padx=(0, 10), pady=10, sticky='e')
        self.label = ctk.CTkLabel(master=self, text='Plot', justify='right')
        self.label.grid(row=1, column=2, padx=(0, 10), pady=10, sticky='e')
        self.label = ctk.CTkLabel(master=self, text='Meta', justify='right')
        self.label.grid(row=1, column=3, padx=(0, 10), pady=10, sticky='e')

        # build
        self.h5rows = []
        self.root_label = None
        self.root = None
        self.build()

    def set_file(self):
        filename = ctk.filedialog.askopenfilename(
            title='Open File',
            filetypes=[('HDF5', '.h5'), ('HDF5', '.hdf5')],
            initialdir=str(Path.cwd()),
        )
        if filename != '':
            self.hdf5_file = filename
            print('new file selecte for viewer')
            self.build()

        else:
            print('no file selected.')

    @property
    def analyzer(self):
        return self.parent.workingtabs.analtoolbar.analyzer

    def go_up(self):
        if self.root is not None:
            with h5py.File(self.hdf5_file, 'r') as f:
                new_root = f[self.root].parent.name
                # print('new root: ', new_root, 'from ', self.root)
            self.build(path=new_root)

    def refresh(self):
        self.build(path=self.root)

    def build(self, path=None):
        for thing in self.h5rows:
            thing.destroy()
        self.h5rows = []

        # make root label
        try:
            self.root_label.destroy()
            self.root_label = None
        except AttributeError:
            pass
        root = path
        self.root = root
        # print('setting_new_root', self.root)
        if root is None:
            root = ''
        self.root_label = ctk.CTkButton(
            master=self,
            text='...' + root + '/',
            anchor='w',
            fg_color='transparent',
            command=self.go_up,
        )
        self.root_label.grid(row=2, column=0, sticky='ew')
        # print('Building viewer from root : ', self.root)
        if os.path.isfile(self.hdf5_file):
            with h5py.File(self.hdf5_file, 'r') as f:
                goto = f
                if path is not None:
                    goto = f[path]
                # print(goto)
                for i, obj in enumerate(goto):
                    o = goto[obj]
                    if isinstance(o, h5py.Group):
                        # print(goto[obj])
                        self.h5rows.append(
                            HDF5GroupRow(i + 3, o, self, 0, hdf5_file=self.hdf5_file)
                        )


def uncertainty_breakdown(file, hdf5_path):
    """
    Break down the uncertainties of a generic RMEMeas object.
    """
    with h5py.File(file, 'r') as f:
        data = load_object(f[hdf5_path], load_big_objects=True)
    # remove a trailing unitary dimension, it's fine
    if len(data.nom.shape) == 2 and data.nom.shape[1] == 1:
        data = data[:, 0]

    if len(data.nom.shape) == 1 and data.nom.dtype is not complex:
        print('1d array, plotting as line')
        fig, ax_budget = plt.subplots(1, 1)

        xlabel = data.nom.dims[0]
        ylabel = hdf5_path.split('/')[-1]
        xvals = data.nom.coords[xlabel]
        # break down uncertainty
        utot = data.stdunc(k=1).cov

        ax_budget.set_ylabel(r'Contributions to Standard Uncertainty (k=1)')
        try:
            data = rmemeas_extras.categorize_by(data, 'Origin')
        except KeyError:
            data = data
        for ploc in data.umech_id:
            unc = data.usel(umech_id=str(ploc)).stdunc()[0]
            ax_budget.plot(xvals, unc, 'o', lw=2, label=ploc)
        ax_budget.plot(xvals, utot, 'ko', lw=2, label='Total')
        box = ax_budget.get_position()
        ax_budget.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax_budget.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax_budget.set_xlabel(xlabel)
        ax_budget.set_ylabel(f'k=1 contribution to {ylabel}')
        ax_budget.set_title(f'Uncertainty in {file}/{hdf5_path}')
        ax_budget.legend(loc='best')
        fig.tight_layout()

    else:
        print('Couldnt plot shape/dtype')
        return None

    return [fig]


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
    k = 2
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

            stdunc = data.stdunc(k=k).cov
            xlabel = data.nom.dims[0]
            ylabel = hdf5_path.split('/')[-1]
            xvals = data.nom.coords[xlabel]
            ax.errorbar(
                xvals,
                data.nom,
                yerr=stdunc,
                fmt='o',
                capsize=3,
                label=f'(k={k}) .../' + '/'.join(hdf5_path.split('/')[-2:]),
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
            stdunc = data.stdunc(k=k).cov
            xlabel = data.nom.dims[0]
            ylabel = hdf5_path.split('/')[-1]
            xvals = data.nom.coords[xlabel]
            ax.errorbar(
                xvals,
                data.nom,
                yerr=stdunc,
                fmt='o',
                capsize=3,
                label=f'(k={k}) .../' + '/'.join(hdf5_path.split('/')[-2:]),
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
                ub = data.uncbounds(k=k)[0]
                lb = data.uncbounds(k=-k)[0]
                xlabel = data_full.nom.dims[0]
                ydim = data_full.nom.dims[1]
                ylabel = ydim + ' : ' + str(data_full.nom.coords[ydim][i].values)
                xvals = data.nom.coords[xlabel]
                ax.plot(
                    xvals,
                    data.nom,
                    'o-',
                    lw=2,
                    label=f'(k={k}).../' + '/'.join(hdf5_path.split('/')[-2:]),
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


def review_s1p(file: str, group: str, plot_style: str = 'lin-mag-phase'):
    """
    Make review plots for an s1p file.

    Parameters
    ----------
    sensor_id : str
        Sensor id of s-parameters to be reviewed.
    plot_style : str, optional
        How to plot the s1p data.
        Options Format
        --------------
        {
        lin-mag-phase
            Linear magnitude and phase (deg)
        real-imag
            Plot as real and imaginary.
        }

    Returns
    -------
    None.

    """

    with h5py.File(file, 'r') as f:
        s1p = RMEMeas.from_h5(f[group])

        k = 2

        prop = RMEProp(sensitivity=True)
        linmag = prop.propagate(vna.linmag)
        phase = prop.propagate(vna.phase)

        @prop.propagate
        def real(a):
            return np.real(a)

        @prop.propagate
        def imag(a):
            return np.imag(a)

        # does nothing, useful for alligning mechanisms
        @prop.propagate
        def align(*a):
            return a

        @prop.propagate
        def minmax(*a):
            stacked = xr.concat(a, dim='repeats')
            mini = stacked.min(dim='repeats')
            maxi = stacked.max(dim='repeats')
            return mini, maxi

        def plot_val(ax, mm, nominal_only=False, color=None, label=None, ls=None):
            ax.plot(mm.nom.frequency, mm.nom, color=color, ls=ls, label=label)
            upper = mm.uncbounds(k=k)[0]
            lower = mm.uncbounds(k=-k)[0]
            if not nominal_only:
                ax.fill_between(
                    mm.nom.frequency,
                    lower[:, 0],
                    upper[:, 0],
                    color=color,
                    alpha=0.2,
                    label=label + ' k = ' + str(k),
                )

        def plot_lin_mag(ax, mm, nominal_only=False, color=None, label=None, ls=None):
            mm = linmag(mm)
            ax.plot(mm.nom.frequency, mm.nom, color=color, ls=ls, label=label)
            upper = mm.uncbounds(k=k)[0]
            lower = mm.uncbounds(k=-k)[0]
            if not nominal_only:
                ax.fill_between(
                    mm.nom.frequency,
                    lower[:, 0],
                    upper[:, 0],
                    color=color,
                    alpha=0.2,
                    label=label + ' k = ' + str(k),
                )

        def plot_phase_diff(
            ax, mm, ref, nominal_only=False, color=None, label=None, ls=None, k=k
        ):
            mm, ref = align(mm, ref)
            mm_p = phase(mm)
            ref_p = phase(ref)

            diff = trig.angle_diff(mm_p.nom, ref_p.nom, deg=False) * 180 / np.pi
            upper = mm_p.stdunc(k=k, rad=True)[0] * 180 / np.pi + diff
            lower = mm_p.stdunc(k=-k, rad=True)[0] * 180 / np.pi + diff
            ax.plot(mm.nom.frequency, np.real(diff), color=color, ls=ls, label=label)

            if not nominal_only:
                ax.fill_between(
                    mm.nom.frequency,
                    lower[:, 0],
                    upper[:, 0],
                    color=color,
                    alpha=0.2,
                    label=label + ' k = ' + str(k),
                )

        def plot_lin_mag_diff(
            ax, mm, ref, nominal_only=False, color=None, label=None, ls=None, k=k
        ):
            mm = linmag(mm)
            ref = linmag(ref)
            mm, ref = align(mm, ref)
            mm.cov -= ref.nom

            ax.plot(mm.nom.frequency, mm.nom, color=color, ls=ls, label=label)
            upper = mm.uncbounds(k=k)[0]
            lower = mm.uncbounds(k=-k)[0]
            if not nominal_only:
                ax.fill_between(
                    mm.nom.frequency,
                    lower[:, 0],
                    upper[:, 0],
                    color=color,
                    alpha=0.2,
                    label=label + ' k = ' + str(k),
                )

        def plot_phase(ax, mm, nominal_only=False, color=None, label=None, ls=None):
            mm = phase(mm) * 180 / np.pi
            ax.plot(mm.nom.frequency, mm.nom, color=color, ls=ls, label=label)
            upper = mm.uncbounds(k=k, deg=True)[0]
            lower = mm.uncbounds(k=-k, deg=True)[0]
            if not nominal_only:
                ax.fill_between(
                    mm.nom.frequency,
                    lower[:, 0],
                    upper[:, 0],
                    color=color,
                    alpha=0.2,
                    label=label + ' k = ' + str(k),
                )

        # plot new value
        if plot_style == 'lin-mag-phase':
            fig, ax = plt.subplots(2, 2, sharex=True)
            ax = ax.transpose()
            plot_lin_mag(ax[0, 0], s1p, label='New', color='b')
            plot_phase(ax[1, 0], s1p, label='New', color='b')

            plot_lin_mag_diff(ax[0, 1], s1p, s1p, label='New', color='b')
            plot_phase_diff(ax[1, 1], s1p, s1p, label='New', color='b')

            ax[0, 0].set_ylabel('Linear Magnitude')
            ax[0, 1].set_ylabel('Phase (deg)')
            ax[0, 1].set_ylabel('Uncertainty on Lin Mag (k=2)')
            ax[1, 1].set_ylabel('Uncertainty on Phase  (k=2)')
            ax[0, 1].set_xlabel('Frequency (GHz)')
            ax[1, 1].set_xlabel('Frequency (GHz)')
        elif plot_style == 'real-imag':
            fig, ax = plt.subplots(1, 2, sharex=True)
            ax = ax.transpose()
            r = real(s1p)
            i = imag(s1p)
            plot_val(ax[0], r, label='New', color='b')
            plot_val(ax[1], i, label='New', color='b')

            ax[0].set_ylabel('Real(S11)')
            ax[1].set_ylabel('Imag(S11)')
            ax[1].set_xlabel('Frequency (GHz)')
            ax[1].set_xlabel('Frequency (GHz)')
        else:
            raise ValueError('plot_style ', plot_style, ' not recognized.')

        fig.suptitle(group.split('/')[-1])

    return fig, ax
