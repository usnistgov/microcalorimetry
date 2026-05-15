# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:27:46 2024

@author: dcg2
"""

import customtkinter as ctk
import json
import os
from pathlib import Path

customtkinter = ctk


class FileFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.filemenu = customtkinter.CTkOptionMenu(
            self, values=['Open', 'Save As'], command=self.filemenu_callback
        )
        self.filemenu.grid(row=0, column=0, padx=(0, 10), sticky='nwe')
        self.filemenu.set('File')
        self._filename = ''

        # button for managing plot settings
        self.plotmenu = customtkinter.CTkOptionMenu(
            self, values=['Export All', 'Close All', 'Pickle All', 'Unpickle'], command=self.plotmenu_callback
        )
        self.plotmenu.grid(row=0, column=1, padx=(0, 10), sticky='nwe')
        self.plotmenu.set('Plots')

    @property
    def graphicstabs(self):
        return self.master.graphicstabs

    @property
    def allforms(self):
        return self.master.workingtabs.forms

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, text):
        self._filename = text
        self.master.title(self.master.title_name + ' : ' + os.path.basename(text))

    def filemenu_callback(self, choice, file=None):
        self.filemenu.set('File')
        if choice == 'Save As':
            self.button_save_as()
        if choice == 'Open':
            self.button_open(file=file)

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
                print(f"Failed to close {tabname} for {e}")
        self.graphicstabs.close_hidden_tabs()

    def plotmenu_export_all(self):
        dialog = customtkinter.CTkInputDialog(
            text='enter a format [.pdf, .png]', title='Export Format'
        )
        text = dialog.get_input()
        folder = str(
            ctk.filedialog.askdirectory(
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
                with open(folder /f'{name}.pklfig', 'wb') as f:
                    pickle.dump(fig, f)
        else:
            print('no folder selected.')
    
    def unpickle_figs(self):
        """Pickles all the open folders to a figure."""
        import pickle
        files =  ctk.filedialog.askopenfilenames(
                title='(Only select files you made, pickling can be unsafe) Select pickle objects to open:',
                filetypes=[('Pickled Figure', '.pklfig')]
            )
        

        for fn in files:
            with open(fn, 'rb') as f:
                fig_loaded = pickle.load(f)
            self.graphicstabs.add_plot(fig_loaded, Path(fn).stem)


    def button_open(self, file=None):
        filename = file
        if filename is None:
            filename = ctk.filedialog.askopenfilename(
                title='Open File', filetypes=[('JSON', '.json')]
            )
        if filename != '':
            with open(filename) as f:
                d = json.load(f)
                # populate tabs
                for k, v in d.items():
                    for frame in self.allforms.values():
                        if k in frame.mytabs.keys():
                            frame.populate_fields(k, v)
            self.filename = filename
            # load data file

        else:
            print('no file selected.')

    def button_save_as(self):
        filename = str(
            ctk.filedialog.asksaveasfilename(
                filetypes=[('JSON', '.json')], title='Save As', defaultextension='.json'
            )
        )

        if filename != '':
            print('collecting for: ')
            print(filename)
            field_dict = {}
            for form in self.allforms.values():
                field_dict.update(form.get_all_field_dicts())
            print(field_dict)
            if os.path.isfile(filename):
                os.remove(filename)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(field_dict, f, ensure_ascii=False, indent=4)
            print(f"File '{filename}' written successfully.")
            self.filename = filename

        else:
            print('no file selected.')
