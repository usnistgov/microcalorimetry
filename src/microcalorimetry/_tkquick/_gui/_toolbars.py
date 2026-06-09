# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:01:26 2024

@author: dcg2
"""

import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
import microcalorimetry._tkquick._gui._stageframes as stageframes
import subprocess as sp
import microcalorimetry._tkquick._gui._forms as forms
import rmellipse.utils
import os
import microcalorimetry._helpers._intf_tools as _intf_tools
import concurrent.futures
import inspect
from pathlib import Path
import time


def helpbutton(tabview, toolbar: 'ClassMethodToolBar'):
    """
    help call back button for a function
    """  # test for class method, if class method call from class
    opentab = toolbar.stagemenu.get()
    # test for class method, if class method call from class
    fun = toolbar.functions[opentab]
    n = fun.__name__
    file_path = inspect.getsourcefile(fun)
    print('')
    print(n)
    print(len(n) * '=')
    print('File Path:', file_path)

    # 2. Get the specific line number where the function is defined
    _, line_no = inspect.getsourcelines(fun)
    print('Line Number:', line_no)
    print('\nSummary')
    print('-------')

    print(fun.__doc__)


def funcrunbutton(tabview, toolbar: 'ClassMethodToolBar'):
    """
    Run callback button for any f

    Parameters
    ----------
    tabview : TYPE
        DESCRIPTION.
    toolbar : TYPE
        DESCRIPTION.

    Raises
    ------
    e
        DESCRIPTION.

    Returns
    -------
    None.

    """

    def add_plots(item):
        if isinstance(item, plt.Figure):
            toolbar.parent.master.parent.graphicstabs.add_plot(item, 'plt')
        elif isinstance(item, list):
            for i in item:
                if isinstance(i, plt.Figure):
                    toolbar.parent.master.parent.graphicstabs.add_plot(i, 'plt')
        elif isinstance(item, rmellipse.utils.GroupSaveable):
            pass

    opentab = toolbar.stagemenu.get()
    pargs, kwargs = tabview.mytabs[opentab]['getargs']()

    # test for class method, if class method call from class
    fun = toolbar.functions[opentab]

    # otherwise call from  instance
    l = len(opentab)
    print('\n ')
    print('*' * l)
    print(opentab)
    print('=' * l)
    print('\nInputs')
    print('------')
    for a in pargs:
        print(a)
    for k in kwargs:
        print(k, ' : ', kwargs[k])
    print('\nAnalysis Log')
    print('------------')
    toolbar.app.update()
    try:
        dryrun = toolbar.dryrun.get()
    except AttributeError:
        dryrun = False
    if not dryrun:
        if fun is not None:
            try:
                output = fun(*pargs, **kwargs)
                # print(output)

                # output = fun(*pargs, **kwargs)
                # print(output)
                if not isinstance(output, tuple):
                    output = (output,)
                # print(output)
                if output is not None:
                    group = toolbar.outputgroup
                    if group is not None:
                        output_path = Path(toolbar.filename_path) / group
                    else:
                        output_path = Path(toolbar.filename_path)
                    if opentab in toolbar.outputs_group_saveable:
                        print('')
                        print('Saving Objects')
                        print('--------------')
                        _intf_tools.save_saveable_objects(
                            output, output_path, function=fun
                        )
                    # try to add plots
                    try:
                        for o in output:
                            add_plots(o)
                    except TypeError:
                        add_plots(output)
            except Exception as e:
                print('FUNCTION FAILED')
                print('---------------')
                toolbar.parent.master.parent.graphicstabs.add_hidden_tabs()
                print('ERROR TRACE BACK')
                print('----------------')
                raise e
        else:
            print(' no analyzer, open a data File.')
    else:
        print('DRY RUN, No function call')
    print('FINISHED')
    print('--------')
    print('\n \n')


class ClassMethodToolBar(ctk.CTkFrame):
    """
    Toolbar for running class methods through a gui.
    """

    def __init__(
        self, master, functions: dict[callable], outputs_group_saveable: list[str]
    ):
        super().__init__(master)
        self.tabview = None
        self.last_save_files = {}
        self.parent = master
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)

        self.stagemenu = None
        self.runbutton = None
        self.dryrun = None

        self.app = self.master.master
        self.functions = functions
        self.outputs_group_saveable = outputs_group_saveable

        self.output_selection_frame = ctk.CTkFrame(self)
        self.output_selection_frame.grid(
            row=1, column=0, columnspan=3, padx=5, pady=5, sticky='nwe'
        )
        self.output_selection_frame.columnconfigure(1, weight=2)
        self.output_selection_frame.columnconfigure(0, weight=0)
        self.file_button = ctk.CTkButton(
            self.output_selection_frame,
            text='save to',
            command=self.button_new_file,
        )

        self.file_button.grid(row=0, column=0, padx=10, pady=(5, 0), sticky='nwe')

        self.filename_entry = ctk.CTkEntry(
            self.output_selection_frame, placeholder_text=None
        )
        self.filename_entry.grid(
            row=0, column=1, padx=(0, 10), pady=(5, 0), sticky='ew', columnspan= 2
        )

        self.group_label = ctk.CTkButton(
            self.output_selection_frame,
            text='under group',
        )
        self.group_label.grid(row=1, column=0, padx=10, pady=(5, 0), sticky='nwe')

        self.group_entry = ctk.CTkEntry(
            self.output_selection_frame, placeholder_text=None
        )
        self.group_entry.grid(row=1, column=1, padx=(0, 10), pady=(5, 0), sticky='ew', columnspan= 2)

        self.app = self.master.master

    @property
    def outputgroup(self):
        val = self.group_entry.get()
        if val == '':
            return None
        return val

    def toggle_output_selection(self, tabview: ctk.CTkTabview):
        produces_outputs = tabview.get() in self.outputs_group_saveable
        # print("toggling tab", produces_outputs)
        if produces_outputs:
            self.output_selection_frame.grid()
        else:
            self.output_selection_frame.grid_remove()

    @outputgroup.setter
    def outputgroup(self, text):
        # print(text)
        self.group_entry.delete(0, last_index=tk.END)
        if text is not None:
            self.group_entry.insert(0, text)
        self.last_save_files[self.tabview.get()]['group'] = text
        # print(self.last_save_files[self.tabview.get()])

    @property
    def filename_path(self):
        return self.filename_entry.get()

    @filename_path.setter
    def filename_path(self, text):
        self.save_to_file = text
        self.last_save_files[self.tabview.get()]['file'] = 'text'
        self.filename_entry.delete(0, last_index=tk.END)
        if text is not None:
            self.filename_entry.insert(0, text)
        # print(self.last_save_files[self.tabview.get()])

    def button_new_file(self):
        filename = str(
            ctk.filedialog.asksaveasfilename(
                filetypes=[('HDF5', '.hdf5'), ('HDF5', '.h5')],
                title='Output File',
                defaultextension='.h5',
                initialdir=str(Path.cwd()),
            )
        )
        if filename != '':
            if os.path.isfile(filename):
                print(f"File '{filename}' selected.")
            self.filename_path = filename
        else:
            print('no file selected.')

    def setup_stage_dropdown(self, tabview, tab_names: list[str]):
        """Set up a drop down menu that switches between different tabs."""

        def optionmenu_callback(choice):
            # save current state
            self.last_save_files[tabview.get()] = {
                'file': self.filename_path,
                'group': self.outputgroup,
            }
            # print('tabview dropdown clicked:', choice)
            tabview.set(choice)
            # print(self.last_save_files)
            if choice in self.last_save_files:
                try:
                    self.filename_path = self.last_save_files[choice]['file']
                    self.outputgroup = self.last_save_files[choice]['group']
                except KeyError:
                    pass
            self.toggle_output_selection(tabview)

        for c in tab_names:
            print('initializing save paths for ', c)
            self.last_save_files[c] = {'file': None, 'group': None}

        self.tabview = tabview
        self.stagemenu = ctk.CTkOptionMenu(
            self, values=tab_names, command=optionmenu_callback
        )
        self.stagemenu.grid(row=0, column=0, padx=10, sticky='new')
        self.last_save_files = {c: {'file': None, 'group': None} for c in tab_names}
        self.toggle_output_selection(tabview)

    def setup_runbutton(self, tabview):
        """Setup a run button that executes the function associated with a given tab."""

        # create a call back passing self to funcrunbutton
        def button_callback():
            return funcrunbutton(tabview, self)

        def help_callback():
            return helpbutton(tabview, self)

        self.runbutton = ctk.CTkButton(self, text='>>', command=button_callback)
        self.runbutton.grid(row=0, column=2, padx=(0, 10), sticky='ne')

        self.helpbutton = ctk.CTkButton(self, text='?', command=help_callback)
        self.helpbutton.grid(row=0, column=1, padx=(0, 10), sticky='ne')
