# -*- coding: utf-8 -*-

import customtkinter as ctk
import inspect
import microcalorimetry._tkquick._gui._themes as themes
from microcalorimetry._tkquick._gui._forms import (
    get_default_args,
    get_form_field,
    ToggleFrame,
)
from numpydoc.docscrape import NumpyDocString
from pathlib import Path
from typing import Callable

customtkinter = ctk


class AnalysisStageFrame(customtkinter.CTkTabview):
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
        doc_frame = customtkinter.CTkFrame(tab, height=100)
        doc_frame.grid(row=0, column=0, sticky='nsew', columnspan=2)
        doc_frame.grid_rowconfigure(0, weight=1)
        for col in [0]:
            doc_frame.grid_columnconfigure(col, weight=1)
        textbox = customtkinter.CTkTextbox(
            doc_frame,
            activate_scrollbars=False,
            font=ctk.CTkFont(**themes.docstring_font),
        )
        textbox.grid(row=0, column=0, sticky='nsew')
        textbox.configure(state='disabled')

        textbox_scrollbar = customtkinter.CTkScrollbar(doc_frame, command=textbox.yview)
        textbox_scrollbar.grid(row=0, column=1, sticky='ns')

        textbox.configure(yscrollcommand=textbox_scrollbar.set)
        textbox.configure(state='normal')
        textbox.insert('end', docstrings)
        textbox.configure(state='disabled')

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
            toggleframe = ToggleFrame(frame, i + 2, 'Fields with Defaults (Optional)')

            for j, p in enumerate(optional_args):
                default = defaults[p.name]
                field = get_form_field(toggleframe.form_frame, j, p, default, level=0)
                fields[p.name] = field

        # function to collect pargs and kwargs
        def getargs():
            kwargs = {}
            for fname, f in fields.items():
                kwargs[fname] = f.get()
            return [], kwargs

        self.mytabs[name] = {
            'funcname': function.__name__,
            'getargs': getargs,
            'fields': fields,
        }

    def get_all_field_dicts(self):
        output = {}
        for tab in self.mytabs:
            print('Seraching fields in ', tab)
            output[tab] = self.get_field_dict(tab)
        return output

    def get_field_dict(self, name):
        output = {}
        for k, v in self.mytabs[name]['fields'].items():
            # print(k,v.get())
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
