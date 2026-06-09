from numpydoc.docscrape import NumpyDocString
from pathlib import Path
from collections import namedtuple
import numpy as np
import customtkinter as ctk
import inspect
import os
import pandas as pd
import microcalorimetry._tkquick.dtypes as dtypes
import microcalorimetry._tkquick._gui._tooltip as _tooltip

customtkinter = ctk

npdoc_typedict = {
    'str': str,
    'float': float,
    'int': int,
    'bool': bool,
    'dict': dict,
    'tuple': tuple,
    'Path': Path,
    'ndarray[float]': np.ndarray[float],
    'np.ndarray[float]': np.ndarray[float],
    'configs.PythonFunction': str,
    'list[Path]': list[Path],
    'Folder': dtypes.Folder,
    'list[int]': list[int],
    'list[str]': list[str],
    'list[float]': list[float],
    'list[configs.EtaHistorical]': list[Path],
    'RFSweepParserConfig': Path,
    'RFSweepSignalConfig': Path,
    'configs.Eta': Path,
    'configs.EtaHistorical': Path,
    'configs.GC': Path,
    'configs.S11': Path,
    'microcalorimetry.configs.ParsedDCSweep': Path,
    'configs.ParsedRFSweep': Path,
    'configs.RFSweep': Path,
    'configs.ThermoelectricFitCoefficients': Path,
}

npdoc_defaults = {
    'str': 'string',
    'float': 'float',
    'int': 'int',
    'bool': False,
    'dict': None,
    'tuple': '0, 0',
    'list': '0, 0',
    'configs.PythonFunction': 'module:function or file.py:function',
    'ndarray[float]': '0.0, 1.0',
    'np.ndarray[float]': '0.0, 1.0',
    'Path': 'Path/To/Thing.ext',
    'RFSweepSignalConfig': 'path.(yml,json,csv)',
    'microcalorimetry.configs.ParsedDCSweep': 'path.(yml,json,csv,h5)',
    'RFSweepParserConfig': 'path.(yml,json,csv)',
    'configs.Eta': 'path.(eff,h5)',
    'configs.EtaHistorical': 'path.(eff,h5)',
    'configs.GC': 'path.(h5)',
    'configs.S11': 'path.(h5,dut)',
    'configs.ParsedRFSweep': 'path.(h5)',
    'configs.RFSweep': 'file.h5/group',
    'configs.ThermoelectricFitCoefficients': 'path.(h5)',
    'list[Path]': 'paths/to/thing.ext, path/to/thing2.ext',
    'list[configs.EtaHistorical]': 'paths/to/thing.yml, path/to/thing.yml',
    'Folder': 'path/to/folder',
    'list[int]': '0, 1, 2, 3',
    'list[str]': 'item1, item2, item3',
    'list[float]': '1.0, 2.0, 3.0',
}

dummy_npparam = namedtuple('dummy_doc', 'name type desc')


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_form_field(master, row, npparam, default, level=0):
    p = npparam
    try:
        fieldname = p.name
        type_str = p.type.split(',')[0]
        dtype = npdoc_typedict[type_str]
    except Exception as e:
        print(npparam)
        raise (e)

    # print(fieldname, type_str, dtype)

    if dtype is str or dtype is float or dtype is int:
        if 'Options Format' in ' '.join(p.desc) and '-' * 14 in ' '.join(p.desc):
            field = DropDownBox(master, row, npparam, default)

        else:
            field = LabelledEntryBox(master, row, p, default=default)

    elif dtype is bool:
        if type(default) is not bool:
            default = 0
        field = BooleanBox(master, row, fieldname, default)

    elif (
        dtype is Path
        or 'Path' in type_str
        or 'Folder' in type_str
        or 'list[configs.EtaHistorical]' in type_str
    ):
        print(type_str, dtype)
        field = PathBox(master, row, fieldname, p.type, default)

    elif dtype is dict:
        field = DictionairyEntry(master, row, p)

    elif dtype is list or dtype is tuple:
        field = SequenceEntry(master, row, p)

    elif 'list' in type_str:
        dtype = npdoc_typedict[type_str.split('[')[1][0:-1]]
        field = SequenceEntry(master, row, p, dtype, default)

    elif 'ndarray' in type_str:
        dtype = npdoc_typedict[type_str.split('[')[1][0:-1]]
        field = NumpyArrayEntry(master, row, p, dtype, default)

    else:
        raise Exception('datatype not recognized')

    # attatche a tool tip
    name = npparam.name
    param_type = npparam.type
    desc = '\n'.join(npparam.desc)

    # Customize your output string format here
    header = '{name}: {param_type}'
    header += '\n' + len(header) * '=' + '\n'
    formatted = f'{desc}'
    _tooltip.CTkToolTip(
        field.label,
        message=formatted,
        justify='left',
    )

    return field


class PathBox:
    def __init__(self, master, row, label, dtype: str, default=None, **kwargs):
        dtype = dtype.split(',')[0]
        self.label = customtkinter.CTkLabel(master, text=label, justify='left')
        self.label.grid(row=row, column=0, padx=(0, 10), pady=10, sticky='w')

        self.field_frame = ctk.CTkFrame(master=master, fg_color='transparent')
        self.field_frame.columnconfigure(0, weight=1)
        self.field_frame.columnconfigure(1, weight=0)
        self.field_frame.grid(row=row, column=1, pady=10, sticky='ew')

        self.box = customtkinter.CTkEntry(
            self.field_frame, placeholder_text=npdoc_defaults[dtype]
        )
        self.box.grid(row=0, column=0, sticky='ew')
        self.button = ctk.CTkButton(
            self.field_frame, text='...', width=20, command=self.fetch_paths
        )
        self.button.grid(row=0, column=1, sticky='e')

        self.dtype = dtype

        if default is not None:
            self.setfield(default)

    def fetch_paths(self):
        print(self.dtype)
        if 'Path' == self.dtype:
            filename = ctk.filedialog.askopenfilename(
                initialdir=str(Path.cwd()), title='Pick File(s)', multiple=False
            )
            if filename != '':
                self.setfield(filename)
                # print(filename)
            else:
                print('no file selected.')

        elif 'list[Path]' == self.dtype:
            filename = ctk.filedialog.askopenfilename(
                title='Pick File(s)',
                multiple=True,
                initialdir=str(Path.cwd()),
            )
            if filename != '':
                filename = str(filename).replace("'", '')[1:-1]
                if filename[-1] == ',':
                    filename = filename[0:-1]
                self.setfield(filename)
                # print(filename)
            else:
                print('no file selected.')

        elif 'Folder' in self.dtype:
            filename = ctk.filedialog.askdirectory(
                title='Pick Folder',
                initialdir=str(Path.cwd()),
            )
            if filename != '':
                self.setfield(filename)
                # print(filename)
            else:
                print('no folder selected.')

        # treat it as a list if list is present
        elif 'list' in self.dtype:
            filename = ctk.filedialog.askopenfilename(
                title='Pick File(s)',
                multiple=True,
                initialdir=str(Path.cwd()),
            )
            if filename != '':
                filename = str(filename).replace("'", '')[1:-1]
                if filename[-1] == ',':
                    filename = filename[0:-1]
                self.setfield(filename)
                # print(filename)
            else:
                print('no file selected.')
        else:
            filename = ctk.filedialog.askopenfilename(
                title='Pick File', multiple=False, initialdir=str(Path.cwd())
            )
            print(filename)
            if filename != '':
                self.setfield(filename)
                # print(filename)
            else:
                print('no file selected.')
            raise Exception(f'{self.dtype} not a recognized pathbox type')

    def get(self):
        text = self.box.get()
        # print(self.label._text, text)
        if text == '':
            return None
        if 'list' in self.dtype:
            return text.replace('"', '').replace("'", '').split(', ')
        else:
            return text

    def setfield(self, text):
        if text is None:
            text = ''
        if self.dtype == 'list[Path]':
            text = str(text).replace('[', '').replace("'", '').replace(']', '')
        self.box.delete('0', last_index='end')
        self.box.insert('1', str(text))


class LabelledEntryBox:
    # not a frame, just organizes a label and an entry box together
    # can be interacted with as a single thing
    def __init__(self, master, row, npparam, default=None, **kwargs):
        label = npparam.name
        type_str = npparam.type.split(',')[0]
        dtype = npdoc_typedict[type_str]

        self.label = customtkinter.CTkLabel(master, text=label, justify='left')
        self.label.grid(row=row, column=0, padx=(0, 10), pady=10, sticky='w')

        self.default_text = default
        self.box = customtkinter.CTkEntry(
            master, placeholder_text=str(npdoc_defaults[type_str])
        )
        self.box.grid(row=row, column=1, sticky='ew')

        self.dtype = dtype

        if default is not None:
            self.setfield(default)

    def get(self):
        text = self.box.get()
        # print(self.label._text, text)
        if text == '':
            return None
        return self.dtype(text)

    def setfield(self, text):
        self.box.delete('0', last_index='end')
        if text is None:
            self.box.configure(textvariable=None)
        else:
            self.box.insert('1', str(text))


class DropDownBox:
    def __init__(self, master, row, npparam, default=None, **kwargs):
        label = npparam.name

        self.label = customtkinter.CTkLabel(master, text=label, justify='left')
        self.label.grid(row=row, column=0, padx=(0, 10), pady=10, sticky='w')

        # build form fields
        desc = npparam.desc
        print('desc is', type(desc))
        i = None
        self.fields = {}
        for di, d in enumerate(desc):
            # print(d)
            if desc[di].lstrip() == 'Options Format':
                i = di + 3
        if i is None:
            raise Exception('Bad Formatting on Options Format Field DOC String')

        values = []
        while desc[i] != '}':
            # make new param
            values.append(desc[i])
            i += 2

        self.box = customtkinter.CTkOptionMenu(master, values=values)
        if str(default) in values:
            self.box.set(str(default))
        elif default is None:
            self.box.set('')
        self.box.grid(row=row, column=1, sticky='ew')

        self.dtype = npdoc_typedict[npparam.type.split(',')[0]]

    def setfield(self, val):
        if val is None:
            val = ''
        self.box.set(str(val))

    def get(self):
        val = self.box.get()
        if val == '':
            return None
        else:
            return self.dtype(val)


class BooleanBox:
    # not a frame, just organizes a label and an entry box together
    # can be interacted with as a single thing
    def __init__(self, master, row, label, default_value, **kwargs):
        self.label = customtkinter.CTkLabel(master, text=label, justify='left')
        self.label.grid(row=row, column=0, padx=(0, 10), pady=10, sticky='w')

        self.box = customtkinter.CTkCheckBox(master, text='')
        self.box.grid(row=row, column=1, sticky='ew', pady=10)

        self.dtype = bool
        self.setfield(default_value)

    def get(self):
        return self.dtype(self.box.get())

    def setfield(self, value):
        check = value
        if type(value) is str:
            check = bool(int(value))
        if check:
            self.box.select()
        else:
            self.box.deselect()


class ToggleFrame(ctk.CTkFrame):
    def __init__(self, master, row, name):
        super().__init__(master)
        self.grid(row=row, column=0, sticky='ew', columnspan=2, pady=10)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # build label
        self.label_frame = ctk.CTkFrame(master=self)
        self.label_frame.grid(row=0, column=0, pady=0, sticky='ew')
        self.label_frame.columnconfigure(0, weight=0)
        self.label_frame.columnconfigure(1, weight=1)

        # form frame
        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.grid(row=1, column=0, pady=0, sticky='ew')
        self.form_frame.columnconfigure(0, weight=0)
        self.form_frame.columnconfigure(1, weight=1)

        # labels
        self.label = ctk.CTkLabel(self.label_frame, text=name)
        self.label.grid(row=0, column=0, sticky='e', padx=(0, 10))

        self.toggle_button = ctk.CTkButton(
            self.label_frame, text='collapse', command=self._toggle_open_close, width=20
        )
        self.toggle_button.grid(row=0, column=1, sticky='ne', padx=0)

    def collapse(self):
        child = self.form_frame
        child.grid_remove()
        self.toggle_button.configure(text='uncollapse')

    def uncollapse(self):
        child = self.form_frame
        child.grid()
        self.toggle_button.configure(text='  collapse')

    def _toggle_open_close(self):
        """
        Open or close the section and change the toggle button image accordingly

        :param ttk.Frame child: the child element to add or remove from grid manager
        """
        child = self.form_frame
        if child.winfo_viewable():
            self.collapse()
        else:
            self.uncollapse()


class DictionairyEntry(ctk.CTkFrame):
    def __init__(self, master, row, npparam):
        super().__init__(master)
        self.grid(row=row, column=0, sticky='ew', columnspan=2, pady=10)
        theme_data = ctk.ThemeManager.theme
        self.configure(fg_color=theme_data['CTk']['fg_color'][1])
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # build label
        self.label_frame = ctk.CTkFrame(master=self)
        self.label_frame.grid(row=0, column=0, pady=0, sticky='ew')
        self.label_frame.columnconfigure(0, weight=0)
        self.label_frame.columnconfigure(1, weight=1)

        # form frame
        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.grid(row=1, column=0, pady=0, sticky='ew')
        self.form_frame.columnconfigure(0, weight=0)
        self.form_frame.columnconfigure(1, weight=1)

        # labels
        self.label = ctk.CTkLabel(self.label_frame, text=npparam.name)
        self.label.grid(row=0, column=0, sticky='e', padx=(0, 10))

        self.toggle_button = ctk.CTkButton(
            self.label_frame,
            text=' collapse',
            command=self._toggle_open_close,
            width=20,
        )
        self.toggle_button.grid(row=0, column=1, sticky='ne', padx=0)

        # build form fields
        desc = npparam.desc
        i = None
        self.fields = {}
        for di, d in enumerate(desc):
            if desc[di] == '{':
                i = di + 1
        if i is None:
            raise Exception('Bad Formatting on Dictionairy Field DOC String')

        dummy_npparam = namedtuple('dummy_doc', 'name type desc')
        row_count = 0
        while desc[i] != '}':
            # make new param
            split = desc[i].split(' : ')
            name = split[0]
            dtype_str = split[1]
            new_desc = [desc[i + 1].lstrip()]
            min_whitespace = len(desc[i + 1]) - len(desc[i + 1].lstrip())
            # search for when leading whitespace get smaller,
            # indicates indent has gone down and we've moved on to the next
            # parameter
            searching = True
            i_search_param = i + 2
            while searching:
                check = desc[i_search_param]
                new_leading_whitespace = len(check) - len(check.lstrip())
                # reach end of parameter description
                if new_leading_whitespace < min_whitespace:
                    searching = False
                else:
                    i_search_param += 1
                    new_desc += [check.lstrip()]

            print(new_desc)
            i = i_search_param

            p = dummy_npparam(name, dtype_str, new_desc)
            default = npdoc_defaults[dtype_str]
            # generate field
            field = get_form_field(self.form_frame, row_count, p, None, level=1)
            row_count += 1
            self.fields[p.name] = field

        self.collapse()

    def get(self):
        return {k: v.get() for k, v in self.fields.items()}

    def setfield(self, setdict):
        for k in setdict:
            self.fields[k].setfield(setdict[k])

    def collapse(self):
        child = self.form_frame
        child.grid_remove()
        self.toggle_button.configure(text='uncollapse')

    def uncollapse(self):
        child = self.form_frame
        child.grid()
        self.toggle_button.configure(text='  collapse')

    def _toggle_open_close(self):
        """
        Open or close the section and change the toggle button image accordingly

        :param ttk.Frame child: the child element to add or remove from grid manager
        """
        child = self.form_frame
        if child.winfo_viewable():
            self.collapse()
        else:
            self.uncollapse()


class NumpyArrayEntry:
    def __init__(self, master, row, param, dtype, default=None, **kwargs):
        label = param.name
        dtype_str = param.type.split(',')[0]

        self.label = customtkinter.CTkLabel(master, text=label, justify='left')
        self.label.grid(row=row, column=0, padx=(0, 10), pady=10, sticky='w')

        self.field_frame = ctk.CTkFrame(master=master, fg_color='transparent')
        self.field_frame.columnconfigure(0, weight=1)
        self.field_frame.columnconfigure(1, weight=0)
        self.field_frame.grid(row=row, column=1, pady=10, sticky='ew')

        self.box = customtkinter.CTkEntry(
            self.field_frame, placeholder_text=npdoc_defaults[dtype_str]
        )
        self.box.grid(row=0, column=0, sticky='ew')
        self.button = ctk.CTkButton(
            self.field_frame, text='...', width=20, command=self.fetch_paths
        )
        self.button.grid(row=0, column=1, sticky='e')

        self.dtype = dtype

        if default is not None:
            self.setfield(default)

    def get(self):
        text = self.box.get()
        print(self.label._text, text)
        if text == '':
            return None
        else:
            # read in a simple csv file if its available
            if os.path.isfile(Path(text)):
                data = (
                    pd.read_csv(Path(text), header=None, index_col=None)
                    .to_numpy()
                    .astype(self.dtype)
                )
                data = data[:, 0]
                print(data, data.shape)
                assert len(data.shape) == 1
                return data
            # otherwise parse box and cast as array
            return np.array([self.dtype(a) for a in text.split(', ')], dtype=self.dtype)

    def setfield(self, ls):
        if ls is None:
            text = ''
        else:
            text = str(ls)
        if self.dtype == str:
            text = text.replace('[', '').replace("'", '').replace(']', '')
        else:
            text = text.replace('[', '').replace(']', '')
        self.box.delete('0', last_index='end')
        self.box.insert('1', text)

    def fetch_paths(self):
        filename = ctk.filedialog.askopenfilename(
            title='Pick Array csv (no header, first column)',
            multiple=False,
            initialdir=str(Path.cwd()),
        )
        if filename != '':
            self.setfield(filename)
            # print(filename)
        else:
            print('no file selected.')


class SequenceEntry:
    # not a frame, just organizes a label and an entry box together
    # can be interacted with as a single thing
    def __init__(self, master, row, param, dtype, default=None, **kwargs):
        label = param.name
        dtype_str = param.type.split(',')[0]

        self.label = customtkinter.CTkLabel(master, text=label, justify='left')
        self.label.grid(row=row, column=0, padx=(0, 10), pady=10, sticky='w')

        self.box = customtkinter.CTkEntry(
            master, placeholder_text=npdoc_defaults[dtype_str]
        )
        self.box.grid(row=row, column=1, sticky='ew')

        self.dtype = dtype

        if default is not None:
            self.setfield(default)

    def get(self):
        text = self.box.get()
        print(self.label._text, text)
        if text == '':
            return None
        else:
            return [self.dtype(a) for a in text.split(', ')]

    def setfield(self, ls):
        if ls is None:
            text = ''
        else:
            text = str(ls)
        if self.dtype == str:
            text = text.replace('[', '').replace("'", '').replace(']', '')
        else:
            text = text.replace('[', '').replace(']', '')
        self.box.delete('0', last_index='end')
        self.box.insert('1', text)
