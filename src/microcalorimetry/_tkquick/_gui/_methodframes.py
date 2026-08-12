# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:03:14 2024

@author: dcg2
"""

from typing import Callable
import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
import subprocess as sp
import microcalorimetry._tkquick._gui._forms as forms
import rmellipse.utils
import os
import microcalorimetry._helpers._intf_tools as _intf_tools
import concurrent.futures
import inspect
from pathlib import Path
import time
import microcalorimetry._tkquick._gui._themes as themes
from microcalorimetry._tkquick._gui._forms import (
    get_default_args,
    get_form_field,
    ToggleFrame,
)
from numpydoc.docscrape import NumpyDocString
import microcalorimetry._tkquick._gui._tooltip as _tooltip
import microcalorimetry._tkquick._gui._history as gui_history

customtkinter = ctk


customtkinter = ctk

# from stagefieldframes import AnalysisStageFieldFrame


class WorkingFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.parent = master


class ModuleTabs(customtkinter.CTkTabview):
    def __init__(self, master, add_scripting_env=False, **kwargs):
        super().__init__(master, **kwargs)
        self.parent = master
        # dictionairy of toolbars for each tab
        self.toolbars: dict[ModuleToolBar] = {}
        # dictionairy of forms that can be accessed by other classes
        self.forms = {}

    @property
    def app(self):
        return self.parent

    def get_function_parameters(
        self, module_tab: str, function_tab: str
    ) -> tuple[dict, str]:
        params, docs = self.toolbars[module_tab].extract_function_parameters(
            function_tab
        )
        return params, docs

    def set_function_parameters(self, module_tab: str, function_tab: str, params: dict):
        self.toolbars[module_tab].set_function_parameters(function_tab, params)

    def open_function(self, module_tab: str, function_tab: str):
        """Open up a function inside a module."""
        self.set(module_tab)
        self.toolbars[module_tab].stagemenu.set(function_tab)
        self.toolbars[module_tab].optionmenu_callback(function_tab)

    def add_module_tab(
        self,
        name: str,
        functions: dict[Callable] = None,
        output_groupsaveable: list[str] = None,
    ):
        """
        Generate a working tab for a module.

        Parameters
        ----------
        name : str
            What to name the tab.
        functions : dict[Callable]
            Dictionary of function objects that can be called.
        output_groupsaveable : list[str]
            List of functions that output groupsaveable objects. These will
            be given the option to select where groupsaveable types can be
            output to.
        """
        # create tabs
        self.add(name)

        # Extras
        # generic functions with no class method.
        self.tab(name).grid_rowconfigure(1, weight=1)
        self.tab(name).grid_columnconfigure(0, weight=1)

        self.toolbars[name] = ModuleToolBar(
            master=self.tab(name),
            functions=functions,
            outputs_group_saveable=output_groupsaveable,
            module_name=name,
        )
        self.toolbars[name].grid(row=0, column=0, padx=0, pady=0)

        extras_lower_frame = WorkingFrame(master=self.tab(name), border_color='black')
        extras_lower_frame.grid(row=1, column=0, sticky='nsew')

        extras_lower_frame.grid_columnconfigure(0, weight=1)
        extras_lower_frame.grid_rowconfigure(0, weight=1)

        extrastages = ModuleFunctionTabsFrame(
            master=extras_lower_frame,
            fg_color='transparent',
            parent=self,
            functions=functions,
            output_groupsaveable=output_groupsaveable,
        )

        extrastages.grid(row=0, column=0, padx=0, pady=0, sticky='nsew')

        self.toolbars[name].connect_to_function_tabs(extrastages)

        self.toolbars[name].setup_function_actions(extrastages)

        # self.toolbars[name].setup_exportbutton(extrastages)

        # self.toolbars[name].setup_importbutton(extrastages)

        # save the form so it can be accessed
        self.forms[name] = extrastages


# -*- coding: utf-8 -*-


class ModuleFunctionTabsFrame(customtkinter.CTkTabview):
    """
    This is a tabbed frame of functions.

    Each tab is named for a function, and contains the fields
    that are input by a user and collected to run that function.

    The actual tabs are hidden, as there may be many which will
    clutter the GUI, so the ModuleFunctionTabsFrame needs to be
    connected to a seperate toolbar that can manage which tab is open via
    a drop down.
    """

    def __init__(
        self,
        master,
        parent,
        functions: dict[Callable],
        output_groupsaveable: list[str],
        **kwargs,
    ):
        super().__init__(master, width=20, **kwargs)
        self.parent = parent

        self.mytabs = {}
        for fn_name, fn in functions.items():
            self.add(fn_name, fn)

        self._outputs_groupsaveable = output_groupsaveable

        # each tab has a 'analysis_function','get_anal_args/kwargs'function
        self._top_spacing = 0
        self._top_button_overhang = 0
        self._segmented_button.grid_forget()
        self._configure_grid()

    def tab_outputs_groupsaveable(self, tabname: str) -> bool:
        return tabname in self._outputs_groupsaveable

    def add(self, name: str, function: Callable):
        """Create an analysis tab derived from a function with Numpy doc strings"""
        super().add(name)

        tab = self.tab(name)

        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # tab.grid(sticky = 'nsew',padx = 0, pady = 0)
        # scrollable frame to see all the stuff
        frame = ctk.CTkScrollableFrame(tab)
        frame.grid(row=1, column=0, sticky='nsew', padx=0, pady=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_rowconfigure(0, weight=1)
        # frame._scrollbar.configure(widt)
        # write the doc strings
        docstrings = function.__doc__

        # doc_frame = customtkinter.CTkFrame(tab, height=100)
        # doc_frame.grid(row=0, column=0, sticky='nsew', columnspan=2)
        # doc_frame.grid_rowconfigure(0, weight=1)
        # for col in [0]:
        #     doc_frame.grid_columnconfigure(col, weight=1)
        # textbox = customtkinter.CTkTextbox(
        #     doc_frame,
        #     activate_scrollbars=False,
        #     font=ctk.CTkFont(**themes.docstring_font),
        # )
        # textbox.grid(row=0, column=0, sticky='nsew')
        # textbox.configure(state='disabled')

        # textbox_scrollbar = customtkinter.CTkScrollbar(doc_frame, command=textbox.yview)
        # textbox_scrollbar.grid(row=0, column=1, sticky='ns')

        # textbox.configure(yscrollcommand=textbox_scrollbar.set)
        # textbox.configure(state='normal')
        # textbox.insert('end', docstrings)
        # textbox.configure(state='disabled')

        # generate fields
        npdocs = NumpyDocString(docstrings)
        defaults = get_default_args(function)
        fields = {}
        optional_args = []

        for i, p in enumerate(npdocs['Parameters']):
            try:
                default = defaults[p.name]
                optional_args.append(p)
            except KeyError:
                default = None

                field = get_form_field(frame, i + 1, p, default, level=0)
                fields[p.name] = field

        if len(optional_args) > 0:
            toggleframe = ToggleFrame(frame, i + 2, 'Override Defaults')

            for j, p in enumerate(optional_args):
                default = defaults[p.name]
                field = get_form_field(
                    toggleframe.form_frame, j, p, default, level=0, is_keyword=True
                )
                fields[p.name] = field

        # function to collect pargs and kwargs
        def getargs():
            kwargs = {}
            for fname, f in fields.items():
                kwargs[fname] = f.get()
            return kwargs

        self.mytabs[name] = {
            'funcname': function.__name__,
            'getargs': getargs,
            'fields': fields,
        }

    def get_all_field_dicts(self):
        output = {}
        for tab in self.mytabs:
            print('Searching fields in ', tab)
            output[tab] = self.get_field_dict(tab)
        return output

    def get_field_dict(self, name):
        output = {}
        for k, v in self.mytabs[name]['fields'].items():
            # print(k,v.get())
            if v.is_active():
                thing = v.get()
                if isinstance(thing, Path):
                    thing = str(thing)
                output[k] = thing
        return output

    def populate_fields(self, name, field_settings):
        try:
            for k, v in self.mytabs[name]['fields'].items():
                try:
                    v.setfield(field_settings[k])
                except KeyError:
                    print(k, ' not a field. May be outdated file.')
        except KeyError:
            print(name, ' tab not found. May be outdated file.')


# %%


class ModuleToolBar(ctk.CTkFrame):
    """
    Toolbar for interacting with functions of a particular module.

    Manages the open function of a ModuleFunctionsTabsFrame, and executes
    actions associated with open functions.
    """

    def __init__(
        self,
        master,
        functions: dict[callable],
        outputs_group_saveable: list[str],
        module_name: str,
    ):
        super().__init__(master)
        self.module_name = module_name
        self.func_tabs = None
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
            row=1, column=0, columnspan=4, padx=5, pady=5, sticky='nwe'
        )
        self.output_selection_frame.columnconfigure(1, weight=2)
        self.output_selection_frame.columnconfigure(0, weight=0)
        self.file_button = ctk.CTkButton(
            self.output_selection_frame,
            text='save output data as',
            command=self.button_new_file,
        )

        self.file_button.grid(row=0, column=0, padx=10, pady=(5, 0), sticky='nwe')

        self.filename_entry = ctk.CTkEntry(
            self.output_selection_frame, placeholder_text=None
        )
        self.filename_entry.grid(
            row=0, column=1, padx=(0, 10), pady=(5, 0), sticky='ew', columnspan=3
        )

        # self.group_label = ctk.CTkButton(
        #     self.output_selection_frame,
        #     text='under group',
        # )
        # self.group_label.grid(row=1, column=0, padx=10, pady=(5, 0), sticky='nwe')

        # self.group_entry = ctk.CTkEntry(
        #     self.output_selection_frame, placeholder_text=None
        # )
        # self.group_entry.grid(
        #     row=1, column=1, padx=(0, 10), pady=(5, 0), sticky='ew', columnspan=2
        # )

        self.app = self.master.master

    # @property
    # def outputgroup(self):
    #     val = self.group_entry.get()
    #     if val == '':
    #         return None
    #     return val
    def optionmenu_callback(self, choice):
        # save current state
        self.last_save_files[self.func_tabs.get()] = {
            'file': self.filename_path,
            # 'group': self.outputgroup,
        }
        # print('func_tabs dropdown clicked:', choice)
        self.func_tabs.set(choice)
        # print(self.last_save_files)
        if choice in self.last_save_files:
            try:
                self.filename_path = self.last_save_files[choice]['file']
                # self.outputgroup = self.last_save_files[choice]['group']
            except KeyError:
                pass
        self.toggle_output_selection(self.func_tabs)

    def connect_to_function_tabs(self, func_tabs: ModuleFunctionTabsFrame):
        """
        Connects the runner toolbar to the different tabs of the functions frame for this module.

        Sets up buttons that interact with the currenty open functional tabs.
        """
        self.func_tabs = func_tabs
        tab_names = list(func_tabs.mytabs.keys())

        for c in tab_names:
            # print('initializing save paths for ', c)
            self.last_save_files[c] = {'file': None}  # , 'group': None}

        self.func_tabs = func_tabs
        self.stagemenu = ctk.CTkOptionMenu(
            self, values=tab_names, command=self.optionmenu_callback
        )
        self.stagemenu.grid(row=0, column=0, padx=10, sticky='new')
        self.last_save_files = {c: {'file': None, 'group': None} for c in tab_names}
        self.toggle_output_selection(func_tabs)

        self.helpbutton = ctk.CTkButton(self, text='?', command=self.help_callback)
        self.helpbutton.grid(row=0, column=1, padx=(0, 10), sticky='ne')

        self.savebutton = ctk.CTkOptionMenu(
            self, values=['save', 'load'], command=self.save_callback
        )
        self.savebutton.grid(row=0, column=2, padx=(0, 10), sticky='ne')
        self.savebutton.set('inputs')

        self.runbutton = ctk.CTkButton(self, text='>>', command=self.run_callback)
        self.runbutton.grid(row=0, column=3, padx=(0, 10), sticky='ne')

    def toggle_output_selection(self, func_tabs: ModuleFunctionTabsFrame):
        produces_outputs = func_tabs.get() in self.outputs_group_saveable
        # print("toggling tab", produces_outputs)
        if produces_outputs:
            self.output_selection_frame.grid()
        else:
            self.output_selection_frame.grid_remove()

    # @outputgroup.setter
    # def outputgroup(self, text):
    #     # print(text)
    #     self.group_entry.delete(0, last_index=tk.END)
    #     if text is not None:
    #         self.group_entry.insert(0, text)
    #     self.last_save_files[self.func_tabs.get()]['group'] = text
    #     # print(self.last_save_files[self.func_tabs.get()])

    @property
    def filename_path(self):
        return self.filename_entry.get()

    @filename_path.setter
    def filename_path(self, text):
        self.save_to_file = text
        self.last_save_files[self.func_tabs.get()]['file'] = text
        self.filename_entry.delete(0, last_index=tk.END)
        if text is not None:
            self.filename_entry.insert(0, text)
        print(self.last_save_files[self.func_tabs.get()])

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

    def set_function_parameters(self, fname: str, config: dict):
        """
        Fill out function's parameters.

        Parameters
        ----------
        fname : str
            Name of the function
        fdict : dict
            dictionary of parameters: {key value pairs of functio inputs} and
            output: path to output file.
        """
        fields = config['parameters']
        output = config['output']
        print(fields)
        # populat input fields
        self.func_tabs.populate_fields(fname, fields)
        # populate output fields

        # if the current open tab matches the function name
        # override the output fields that are active
        if self.stagemenu.get() == fname:
            self.filename_path = output

        # save the dictionary of output fields as well
        self.last_save_files[fname]['file'] = output

    def extract_function_parameters(self, fname: str) -> tuple[dict, str]:
        """
        Get the parameters currently used in a function.

        Parameters
        ----------
        fname : str
            None

        Returns
        -------
        tuple[dict, str]
            _description_
        """
        opentab = self.stagemenu.get()
        kwargs = self.func_tabs.mytabs[opentab]['getargs']()
        function = self.functions[opentab]
        docs = function.__doc__

        # collect form values
        field_dict = self.func_tabs.get_field_dict(opentab)

        from importlib.metadata import version

        # collect meatadata
        fname = function.__name__
        version = version('microcalorimetry')

        # collect outputpath
        output_path = self.filename_path
        if output_path == '' or output_path is None:
            output_path = None
        else:
            try:
                output_path = Path().relative_to(Path.cwd())
            except ValueError:
                output_path = Path(output_path)

            output_path = output_path.as_posix()

        data = {
            'microcalorimetry': version,
            'guimodule': self.module_name,  # name of the tab in the GUI
            'guifunction': opentab,  # name of the function in the GUI
            'pyname': function.__name__,  # name of the function in Python
            'pymodule': function.__module__,  # name of the module in Python
            'output': output_path,  # groupsaveable serialized data outputs path
            'parameters': field_dict,  # function parameters
        }
        return data, docs

    # create a call back passing self to funcrunbutton

    def help_callback(self):
        """
        help call back button for a function
        """
        toolbar = self
        func_tabs = self.func_tabs

        # test for class method, if class method call from class
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

    def export_callback(self):
        """
        help call back button for a function
        """
        # test for class method, if class method call from class
        toolbar = self
        func_tabs = self.func_tabs

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

    def run_callback(self):
        """
        Run a function using current inputs of the active function.

        """
        toolbar = self
        func_tabs = self.func_tabs

        # add to history file

        def add_plots(item):
            print('adding plots')
            if isinstance(item, plt.Figure):
                toolbar.parent.master.parent.graphicstabs.add_plot(item, 'plt')
            elif isinstance(item, list):
                for i in item:
                    if isinstance(i, plt.Figure):
                        toolbar.parent.master.parent.graphicstabs.add_plot(i, 'plt')
            elif isinstance(item, rmellipse.utils.GroupSaveable):
                pass

        opentab = toolbar.stagemenu.get()
        parameters, _ = self.extract_function_parameters(opentab)
        filepath = parameters['output']
        kwargs = parameters['parameters']

        # test for class method, if class method call from class
        fun = toolbar.functions[opentab]

        gui_history.append_history(
            fname=opentab, module=self.module_name, parameters=parameters
        )

        # otherwise call from  instance
        l = len(opentab)
        print('\n ')
        print('*' * l)
        print(opentab)
        print('=' * l)
        print('\nInputs')
        print('------')
        # for a in pargs:
        #     print(a)
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
                    output = fun(**kwargs)
                    # print(output)

                    # output = fun(*pargs, **kwargs)
                    # print(output)
                    if not isinstance(output, tuple):
                        output = (output,)
                    # print(output)
                    if output is not None:
                        # stream dataset outputs to an hdf5 file
                        if filepath is not None:
                            group = None
                            if group is not None:
                                output_path = Path(filepath) / group
                            else:
                                output_path = Path(filepath)
                            if opentab in toolbar.outputs_group_saveable:
                                print('')
                                print('Saving Objects')
                                print('--------------')
                                _intf_tools.save_saveable_objects(
                                    output, output_path, function=fun
                                )
                        # try to add plots to the graphics
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

    def save_callback(self, value):
        self.savebutton.set('inputs')
        match value:
            case 'save':
                opentab = self.stagemenu.get()
                # collect save target
                filename = str(
                    ctk.filedialog.asksaveasfilename(
                        initialfile=opentab + '.yml',
                        filetypes=[
                            # ('Experiment Parameters','.csv'),
                            ('YAML', '.yml'),
                            # ('JSON', '.json')
                        ],
                        title=f'Save inputs for {opentab}',
                        # defaultextension='.csv',
                        initialdir=str(
                            Path.cwd(),
                        ),
                    )
                )

                filename = Path(filename)
                print('saving to...', filename)
                extension = filename.suffix
                data, docs = self.extract_function_parameters(opentab)

                match extension:
                    # case '.csv':
                    #     from rminstr.data_structures import ExptParameters
                    #     params = ExptParameters([], initial_dict=field_dict)
                    #     print(params)
                    #     params.save_config(filename.as_posix())
                    case '.yml':
                        from yaml import safe_dump

                        with open(filename, 'w') as f:
                            f.write('# Python Function Documentation\n')
                            f.write('# =============================\n')
                            for line in docs.splitlines():
                                f.write('# ' + line + '\n')
                            safe_dump(data, f, indent=True, sort_keys=False)

                    case _:
                        raise ValueError('Extension not recognized.')

            case 'load':
                opentab = self.stagemenu.get()
                kwargs = self.func_tabs.mytabs[opentab]['getargs']()
                filename = str(
                    ctk.filedialog.askopenfilename(
                        filetypes=[
                            # ('Experiment Parameters','.csv'),
                            ('YAML', '.yml'),
                            ('JSON', '.json'),
                        ],
                        title=f'Load inputs for {opentab}',
                        # defaultextension='.csv',
                        initialdir=str(Path.cwd()),
                    )
                )
                filename = Path(filename)
                print('loading from...', filename)
                extension = filename.suffix

                # collect form values
                match extension:
                    # case '.csv':
                    #     from rminstr.data_structures import ExptParameters
                    #     fields = ExptParameters(filename)
                    case '.yml':
                        from yaml import safe_load

                        with open(filename, 'r') as f:
                            data = safe_load(f)
                    # case '.json':
                    #     import json
                    #     with open(filename, 'r') as f:
                    #         fields = json.load(f)
                    case _:
                        raise ValueError('Extension not recognized.')
                self.set_function_parameters(opentab, data)

            case _:
                print('Option not recognized')
        # print(value)
        return

    def setup_function_actions(self, func_tabs):
        """Setup a run button that executes the function associated with a given tab."""
