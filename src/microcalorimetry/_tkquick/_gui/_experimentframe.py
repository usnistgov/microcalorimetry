# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:58:51 2025

@author: dcg2
"""

import customtkinter as ctk
import microcalorimetry._tkquick._gui._forms as forms
import subprocess as sp


class ExptFrames(ctk.CTkTabview):
    def __init__(self, master, parent, **kwargs):
        super().__init__(master, width=20, **kwargs)
        self.parent = parent
        self.mytabs = {}
        self.setup_sample_experiment()
        self.setup_calorimeter_run()
        self.setup_sensitivity()

        # each tab has a 'analysis_function','get_anal_args/kwargs'function
        self._top_spacing = 0
        self._top_button_overhang = 0
        self._segmented_button.grid_forget()
        self._configure_grid()

    def setup_sample_experiment(self):
        name = 'testexp'
        tab = self.add(name)
        # set up analysis mode
        tab.grid_columnconfigure(0, weight=1)
        self.mytabs[name] = {}
        fields = {}
        # setup
        fields['--output-dir'] = forms.PathBox(tab, 0, 'output-dir', 'Folder')

        fields['--repeats'] = forms.LabelledEntryBox(
            tab,
            3,
            forms.dummy_npparam('repeats', 'int', 'number of iterations'),
        )

        fields['--delay'] = forms.LabelledEntryBox(
            tab,
            4,
            forms.dummy_npparam('delay', 'float', 'delay time in seconds'),
        )

        # save stuff
        self.mytabs[name]['fields'] = fields

    def setup_sensitivity(self):
        name = 'sensitivity'
        tab = self.add(name)
        tab.grid_columnconfigure(1, weight=1)
        self.mytabs[name] = {}
        fields = {}
        fields['--output-dir'] = forms.PathBox(tab, 0, 'output-dir', 'Folder')

        fields['--settings'] = forms.PathBox(tab, 1, 'settings', 'Path')

        fields['--measlist'] = forms.PathBox(tab, 2, 'measlist', 'Path')

        # save stuff
        self.mytabs[name]['fields'] = fields

    def setup_calorimeter_run(self):
        name = 'calibrate'
        tab = self.add(name)
        tab.grid_columnconfigure(1, weight=1)
        self.mytabs[name] = {}
        fields = {}
        fields['--output-dir'] = forms.PathBox(tab, 0, 'output-dir', 'Folder')

        fields['--configs'] = forms.PathBox(tab, 2, 'configs', 'list[Path]')

        fields['--settings'] = forms.PathBox(tab, 3, 'settings', 'Path')

        fields['--repeats'] = forms.LabelledEntryBox(
            tab,
            4,
            forms.dummy_npparam(
                'repeats', 'int', 'number of times to repeat experiment'
            ),
        )

        fields['--name'] = forms.LabelledEntryBox(
            tab,
            5,
            forms.dummy_npparam('name', 'str', 'name of folder to store measurement.'),
        )

        # save stuff
        self.mytabs[name]['fields'] = fields

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
            output[k] = v.get()
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
