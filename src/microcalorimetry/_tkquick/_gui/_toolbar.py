# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:27:46 2024

@author: dcg2
"""

import customtkinter as ctk
import json
import os
from pathlib import Path
from microcalorimetry._tkquick._gui._history import HistoryTopLevel


class ToolBarFrame(ctk.CTkFrame):
    """Defines the tool bar at the top of the GUI."""

    def __init__(self, master):
        super().__init__(master)

        self.columnconfigure(2, weight=2)
        self.columnconfigure((0, 1), weight=0)

        # working directory label
        self.cwd_label = ctk.CTkLabel(self)
        self.update_cwd_label()
        self.cwd_label.grid(row=0, column=10, padx=(0, 10), sticky='ew')

        self.filemenu = ctk.CTkOptionMenu(
            self,
            values=['Change Working Directory', 'History'],
            command=self.filemenu_callback,
        )
        self.filemenu.grid(row=1, column=0, padx=(0, 10), sticky='nwe')
        self.filemenu.set('GUI')
        self._filename = ''

        # button for managing plot settings
        self.plotmenu = ctk.CTkOptionMenu(
            self,
            values=['Export All', 'Close All', 'Pickle All', 'Unpickle'],
            command=self.plotmenu_callback,
        )
        self.plotmenu.grid(row=1, column=1, padx=(0, 10), sticky='nwe')
        self.plotmenu.set('Plots')

    def update_cwd_label(self):
        self.cwd_label.configure(text='Working Directory: ' + Path.cwd().as_posix())

    @property
    def graphicstabs(self):
        return self.master.graphicstabs

    @property
    def allforms(self):
        return self.master.moduletabs.forms

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, text):
        self._filename = text
        self.master.title(self.master.title_name + ' : ' + os.path.basename(text))

    def history_callback(self):
        history = HistoryTopLevel()

    def filemenu_callback(self, choice, file=None):
        self.filemenu.set('GUI')
        match choice:
            case 'Change Working Directory':
                self.change_dir()
            case 'History':
                self.history_callback()
            case _:
                raise ValueError(f'{choice} not recognized')

    def plotmenu_callback(self, choice, file=None):
        self.plotmenu.set('Plots')
        match choice:
            case 'Export All':
                self.plotmenu_export_all()
            case 'Close All':
                self.plotmenu_close_all()
            case 'Pickle All':
                self.pickle_all_figs()
            case 'Unpickle':
                self.unpickle_figs()
            case _:
                raise Exception(f'{choice} not recognized')

    def plotmenu_close_all(self):
        tabnames = list(self.graphicstabs.plots_dict.keys())
        for tabname in tabnames:
            try:
                self.graphicstabs.delete(tabname)
            except Exception as e:
                print(f'Failed to close {tabname} for {e}')
        self.graphicstabs.close_hidden_tabs()

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
            plots_dict = self.graphicstabs.plots_dict
            for name, fig in plots_dict.items():
                fig.savefig(folder / f'{name}{frmt}')
        else:
            print('no folder selected.')

    def pickle_all_figs(self):
        """Pickles all the open folders to a figure."""
        import pickle

        folder = str(
            ctk.filedialog.askdirectory(
                title='Pickle open figures to folder:',
            )
        )

        if folder != '':
            folder = Path(folder)
            plots_dict = self.graphicstabs.plots_dict
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
            self.graphicstabs.add_plot(fig_loaded, Path(fn).stem)

    def change_dir(self):
        newdir = ctk.filedialog.askdirectory(
            title='Pick New Working Directory', initialdir=str(Path.cwd())
        )
        if newdir is not None:
            os.chdir(Path(newdir))
            print(f'changed to {newdir}')
            self.update_cwd_label()
