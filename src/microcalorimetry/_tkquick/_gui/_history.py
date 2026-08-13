from microcalorimetry._tkquick.settings import cwd_history_file
from datetime import datetime
from pathlib import Path
from fnmatch import fnmatch
import yaml
import customtkinter as ctk
import shutil
from typing import TypedDict, TypeAlias
from functools import partial
from microcalorimetry._tkquick.settings import get_cwd_settings


class FunctionSaveDict(TypedDict):
    """Version of microcalorimetry."""

    microcalorimetry: str
    """Name of GUI module tab."""
    guimodule: str
    """Name of GUI function tab."""
    guifunction: str
    """Name of Python function."""
    pyname: str
    """Name of Python module containing the function."""
    pymodule: str
    """Output path."""
    output: str
    """Function inputs as key value pairs"""
    parameters: dict


def today() -> str:
    return datetime.now().strftime('%Y%m%d')


def clear_history():
    """Clear history file."""
    Path(cwd_history_file()).unlink(missing_ok=True)


def read_history() -> dict[FunctionSaveDict]:
    try:
        with open(cwd_history_file(), 'r') as f:
            history = yaml.safe_load(f)
        if history is None:
            return {}
    except FileNotFoundError:
        history = {}
    return history


def append_history(fname: str, module: str, parameters: FunctionSaveDict):
    """
    Append a function parameter set to the history file.

    Parameters
    ----------
    fname : str
        Name that will be added to the field.
    module : str
        Name that will be added to the field.
    parameters : dict
        _description_
    """
    history = read_history()

    field_name_base = f'{module}.{fname}-{today()}'
    done = False
    count = 0
    while not done:
        fieldname = f'{field_name_base}-{count}'
        # print(fieldname)
        if fieldname not in history:
            done = True
        count += 1

    history[fieldname] = parameters

    with open(cwd_history_file(), 'w') as f:
        yaml.safe_dump(history, f, indent=True, sort_keys=False)


class HistoryTopLevel(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # grab
        from microcalorimetry._tkquick._api import _APP, GUI

        self.app: GUI = _APP
        self.title('microcalorimetry history')
        self.geometry('700x500')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        # self.focus()

        # history filters and such
        self.refresh_button = ctk.CTkButton(
            self, text='refresh', width=100, command=self.refresh_callback
        )
        self.refresh_button.grid(row=0, column=1)

        self.glob_entry = ctk.CTkComboBox(
            self, values=[self.default_glob], command=self.refresh_callback
        )
        self.glob_entry.grid(row=0, column=0, sticky='ew')

        # filter and view function calls that have been saved to history
        self.histlist = HistoryList(self)
        self.histlist.grid(row=1, column=0, columnspan=2, sticky='nesw')
        # place initial labels

        self.histlist.place_labels(self.glob)

        # display options after selecting a function
        self.options = HistoryOptions(self)
        self.options.grid(row=1, column=2, sticky='nsew')

        # Bring to front temporarily
        self.lift()
        self.attributes('-topmost', True)
        self.focus()

    @property
    def default_glob(self) -> str:
        return f'*{today()}*'

    @property
    def glob(self) -> str:
        val = self.glob_entry.get()
        if val == '' or val is None:
            val = self.default_glob
        return val

    def refresh_callback(self, new_glob=None):
        # print('Refreshing called')
        if new_glob is None:
            new_glob = self.glob
        self.histlist.place_labels(new_glob)
        # update combo box so it remembers your filters.
        old_globs = self.glob_entry.cget('values')
        if new_glob in old_globs:
            old_globs.remove(new_glob)
        new_options = [new_glob] + old_globs
        self.glob_entry.configure(values=new_options)

    def update_options_label(self, label: str):
        self.options.update_label(label)


class HistoryOptions(ctk.CTkFrame):
    """Displays selected function and provides options."""

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text='(select a function from the list)')
        self.label.grid(row=0, sticky='ew', padx=10)

        self.load_button = ctk.CTkButton(
            self,
            text='load to GUI (overwrites current entries)',
            command=self.load_callback,
        )
        self.load_button.grid(row=1, stick='ew', padx=10)
        self.toplevel = master
        self.app = self.toplevel.app

    def update_label(self, name: str):
        self.label.configure(text=name)

    def load_callback(self):
        field_name = self.label.cget('text')
        params = self.toplevel.histlist.values[field_name]
        print(f'loading {field_name}...')
        import yaml

        module = params['guimodule']
        function = params['guifunction']

        # load parameters and open up the new tab

        self.app.moduletabs.set_function_parameters(
            module_tab=module, function_tab=function, params=params
        )

        self.app.moduletabs.open_function(module, function)


class Row(TypedDict):
    """Row of a HistoryList."""

    button: ctk.CTkButton


class HistoryList(ctk.CTkScrollableFrame):
    """Sorts and displays function calls you've made."""

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.values: dict = {}
        self.rows: list[Row, ...] = []
        self.columnconfigure(0, weight=1)

    def button_callback(self, index: int):
        selection = self.get_label(index)
        print(f'Field name selected {selection}')
        self.master.update_options_label(selection)

    def get_label(self, index: int):
        return self.rows[index]['button'].cget('text')

    def make_row(self, index: int, label: str):
        # button call back saves the row index, which
        # never changes, and the label can be retrievd
        # when the function is called
        new_button = ctk.CTkButton(
            self,
            text=label,
            fg_color='transparent',
            command=partial(self.button_callback, index),
        )
        new_button.grid(row=index, column=0, sticky='w')
        row = {'button': new_button}
        self.rows.append(row)

    def destroy_row(self, index):
        for item in self.rows[index].values():
            # print(item)
            item.destroy()

    def rename_label(self, index: int, new_name: str):
        self.rows[index]['button'].configure(text=new_name)

    def place_labels(self, glob: str):
        # print(f"placing labels {glob}")
        history = read_history()
        history = {k: v for k, v in history.items() if fnmatch(k, glob)}

        # cache the history values
        self.values = history
        # print(f"Filtered to {len(history)} function calls")

        # reverse the names so newest entries are on top
        names = list(history.keys())[::-1]

        # modify new labels OR place new labels
        for i, name in enumerate(names):
            if i >= len(self.rows):
                self.make_row(i, name)
            else:
                self.rename_label(i, name)

        # delte extra labels
        while len(self.rows) > len(names):
            self.destroy_row(len(self.rows) - 1)
            self.rows.pop()
