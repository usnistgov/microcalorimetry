# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:03:14 2024

@author: dcg2
"""

from microcalorimetry._tkquick._gui._stageframes import AnalysisStageFrame
from microcalorimetry._tkquick._gui._toolbars import ClassMethodToolBar
from typing import Callable
import customtkinter as ctk

customtkinter = ctk

# from stagefieldframes import AnalysisStageFieldFrame


class WorkingFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.parent = master


class WorkingTabs(customtkinter.CTkTabview):
    def __init__(self, master, add_scripting_env=False, **kwargs):
        super().__init__(master, **kwargs)
        self.parent = master
        # dictionairy of toolbars for each tab
        self.toolbars = {}
        # dictionairy of forms that can be accessed by other classes
        self.forms = {}

    def add_module_tab(
        self,
        name: str,
        functions: dict[Callable] = None,
        output_groupsaveable: list[str] = None,
    ):
        """
        Generate a working tab for a module.

        Fuinctions added are inferred from the __all__ attribute of the module.

        Parameters
        ----------
        name : str
            What to name the tab.
        functions : dict[Callable]
            Dictionary of function objects that can be called.
        output_groupsaveable : list[str]
            List of functions that output a groupsaveable method, and should
            display an output selection, and parse outputs for group_saveable objects.
        """
        # create tabs
        self.add(name)

        # Extras
        # generic functions with no class method.
        self.tab(name).grid_rowconfigure(1, weight=1)
        self.tab(name).grid_columnconfigure(0, weight=1)

        self.toolbars[name] = ClassMethodToolBar(
            master=self.tab(name),
            functions=functions,
            outputs_group_saveable=output_groupsaveable,
        )
        self.toolbars[name].grid(row=0, column=0, padx=0, pady=0)

        extras_lower_frame = WorkingFrame(master=self.tab(name), border_color='black')
        extras_lower_frame.grid(row=1, column=0, sticky='nsew')

        extras_lower_frame.grid_columnconfigure(0, weight=1)
        extras_lower_frame.grid_rowconfigure(0, weight=1)

        extrastages = AnalysisStageFrame(
            master=extras_lower_frame,
            fg_color='transparent',
            parent=self,
            functions=functions,
            output_groupsaveable=output_groupsaveable,
        )

        extrastages.grid(row=0, column=0, padx=0, pady=0, sticky='nsew')

        self.toolbars[name].setup_stage_dropdown(
            extrastages, list(extrastages.mytabs.keys())
        )
        self.toolbars[name].setup_runbutton(extrastages)

        # save the form so it can be accessed
        self.forms[name] = extrastages
