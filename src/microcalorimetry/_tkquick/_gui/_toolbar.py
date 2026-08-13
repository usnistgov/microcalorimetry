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
from microcalorimetry._tkquick.settings import get_user_settings


class CWDNavigator(ctk.CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.columnconfigure(1, weight=1)
        self.label = ctk.CTkButton(
            self, text='CWD', command=self.label_callback, fg_color='transparent'
        )
        self.label.grid(row=0, column=0, padx=(0, 10))
        self.button = ctk.CTkSegmentedButton(
            self, values=[], command=self.nav_callback, width=10
        )
        self.button.grid(row=0, column=1, sticky='ew')

    def update(self):
        parents = Path.cwd().parents[::-1]
        parents = [Path(p).name + '/' for p in parents]
        stem = Path.cwd().name
        full_path = [Path.cwd().drive + '/'] + list(parents)[1:] + [stem]
        self.button.configure(values=full_path)

        # store in settings
        user_settings = get_user_settings()
        user_settings.last_cwd = Path.cwd().as_posix()

        # deselects buttons
        self.button.set('')

    def label_callback(self):
        newdir = ctk.filedialog.askdirectory(
            title='Pick a new working directory.', initialdir=Path.home()
        )
        if newdir is not None:
            os.chdir(Path(newdir))
            print(f'changed to {newdir}')
            self.update()

    def get(self):
        return Path.cwd()

    def nav_callback(self, choice):
        values = self.button.cget('values')
        i = [i for i in range(len(values)) if values[i] == choice][0]
        path = '/'.join(values[0 : i + 1])

        newdir = ctk.filedialog.askdirectory(
            title='Pick a new working directory.', initialdir=Path(path).resolve()
        )

        if newdir is not None:
            os.chdir(Path(newdir))
            print(f'changed to {newdir}')
            self.update()


class ToolBarFrame(ctk.CTkFrame):
    """Defines the tool bar at the top of the GUI."""

    def __init__(self, master):
        super().__init__(master)

        self.columnconfigure(0, weight=1)
        self.columnconfigure((0, 1), weight=1)

        # working directory label
        self.cwd_frame = ctk.CTkFrame(self, fg_color='transparent')
        self.cwd_frame.grid(row=0, sticky='ew')

        self.cwd_label: CWDNavigator = CWDNavigator(self.cwd_frame)
        self.cwd_label.grid(row=0, column=0, padx=(0, 10), sticky='ew')
        self.cwd_label.update()
