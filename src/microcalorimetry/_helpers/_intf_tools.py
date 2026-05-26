import click
import importlib
import inspect
import matplotlib as mpl
from numpydoc.docscrape import FunctionDoc as _scrape
from rmellipse.utils import save_object, GroupSaveable
import rmellipse.utils._group_saveable as _group_saveable
from pathlib import Path
from numpydoc.docscrape import FunctionDoc
from typing import Iterable
import matplotlib.pyplot as plt
import h5py
import io
import microcalorimetry.configs as configs
import sys
import os
import numpydoc
import warnings

_group_saveable.SAVEABLE


class LazyGroup(click.Group):
    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        # lazy_subcommands is a map of the form:
        #
        #   {command-name} -> {module-name}.{command-object-name}
        #
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        lazy = sorted(self.lazy_subcommands.keys())
        return base + lazy

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name):
        # lazily loading a command, first get the module name and attribute name
        import_path = self.lazy_subcommands[cmd_name]
        modname, cmd_object_name = import_path.rsplit('.', 1)
        # do the import
        mod = importlib.import_module(modname)
        # get the Command object from that module
        cmd_object = getattr(mod, cmd_object_name)
        # check the result to make debugging easier
        if not isinstance(cmd_object, click.Command):
            raise ValueError(
                f'Lazy loading of {import_path} failed by returning '
                'a non-command object'
            )
        return cmd_object


def recurse_for(thing: Iterable, search_type: object) -> object:
    if isinstance(thing, search_type):
        yield thing
    elif isinstance(thing, dict):
        for child_d in thing.values():
            yield from recurse_for(child_d, search_type)
    elif isinstance(thing, list) or isinstance(thing, tuple):
        for child in thing:
            yield from recurse_for(child, search_type)


def run_and_show_plots(
    callable: object,
    *callable_args,
    show_plots: bool,
    save_plots: Path = None,
    plot_ext: str = '.png',
    **callable_kwargs,
):
    if show_plots:
        mpl.use('qtagg')
    outputs = callable(*callable_args, **callable_kwargs)
    if save_plots:
        for o in recurse_for(outputs, plt.Figure):
            o.savefig(Path(save_plots) / (str(o.number) + plot_ext))
    return outputs


def save_saveable_objects(
    objects: tuple[object], output_file: Path, function: callable = None
):
    """
    Save any saveable objects output from a function with groupsaveable.

    Parameters
    ----------
    objects : tuple[object]
        Objects.
    output_file : Path
        File to output to, supports extending path with group. e.g.
        file.h5/group.
    function : callable, optional
        If provided, utilizes function to name the outputs.

    Returns
    -------
    None.

    """
    # Save data, only overwrite existing data if asked to
    path, group = configs.split_h5path(output_file)

    # don't save if no path or point to a directory.
    if path is None or Path(path).is_dir():
        print('Valid save path not provided. Not saving outputs.')
        return

    # if callable was provided, inspect for names
    # of the return parameters.
    object_names = []
    if function is not None:
        try:
            return_params = FunctionDoc(function)['Returns']
            object_names = [p.name for p in return_params]
        except Exception as e:
            warnings.warn(
                f"""
                Failed to parse doc string due to error on save saveable object: {e}
                """
            )
            object_names = []

        for i, o in enumerate(object_names):
            if not o.isidentifier():
                warnings.warn(
                    f'return parameter: "{o}"  of {function.__name__} is not a valid identifier. Modify doc strings to a valid identifier.'
                )
                object_names[i] = None

    # iterate over objects dump into an hdf5 file.
    for i, o in enumerate(objects):
        is_groupsaveable = isinstance(o, _group_saveable.SAVEABLE) or isinstance(
            o, GroupSaveable
        )
        is_list = isinstance(o, list) or isinstance(o, tuple)
        list_of_figs = False

        # some edge cases of things I think arent worth saving.
        # ignore iterables of figures, they can't be saved and group
        # saveable will skip over the figure elements and save empty
        # lists rather then throw an error.
        if not (is_groupsaveable or is_list):
            print(f'Not saving {type(o)}, not group saveable')
            continue
        elif is_list and len(o) == 0:
            print(f'Not saving {type(o)}, empty list')
            continue
        elif is_list and isinstance(o[0], plt.Figure):
            print(f'Not saving {type(o)}, list of figures')
            continue
        elif is_list and all([oi is None for oi in o]):
            print(f'Not saving {type(o)}, List of None')
            continue
        elif o is None:
            print(f'Not saving {None}, None')
            continue



        with h5py.File(path, 'a') as f:
            g = f
            if group is not None:
                g = f.require_group(group)

            # start with n
            name = None
            try:
                name = object_names[i]
            except IndexError:
                pass
            if name is None:
                # no name was defined in docs
                # fall back to name assigned by group saveable
                try:
                    name = object.name
                # no name in group saveable, fall back to
                # class type name.
                except AttributeError:
                    name = type(o).__name__

            if name in g:
                count = 0
                while name + f'_{count}' in g:
                    count += 1
                name = name + f'_{count}'

            try:
                save_object(g, name, o)
                print('Saved ', name)
                print('  type  :', type(o))
                print('  path  :', str(Path(path).as_posix()) + f'{g.name}/{name}')
                print('')

            except Exception as e:
                warnings.warn(f'failed to save {type(o)} for {e}')
                del g[name]


def format_from_npdoc(function: any) -> callable:
    """
    Generate a function that formats a CLICK command.

    Returns a callable that uses the signature/docstrings
    of function to format a CLICK command.

    Assigns defaults from the function signature, and the
    first line of the Parameter description to the helpstring.
    Assumes CLI parameters are the same name as the function
    signature/doc strings, replaces '-' in parameter names
    with an underscore.

    Parameters
    ----------
    function : any
        Function containing the signature/docstring

    Returns
    -------
    callable
        New function that formats a click Command.
    """

    def inner(command: click.Command):
        # scrape signature and doc strings
        assert isinstance(command, click.Command)
        # get things defined in the original function
        doc = _scrape(function)
        func_summary = '\n'.join(doc['Summary'] + doc['Extended Summary'])
        func_param_docs = doc['Parameters']
        func_param_docs = {p.name: p.desc for p in func_param_docs}
        func_signature = inspect.signature(function)
        func_defaults = {
            k: v.default
            for k, v in func_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        # get anything extra defined in the interface function
        doc = _scrape(command)

        cmnd_param_docs = doc['Parameters']
        cmnd_param_docs = {p.name: p.desc for p in cmnd_param_docs}
        cmnd_signature = inspect.signature(function)
        cmnd_defaults = {
            k: v.default
            for k, v in cmnd_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        # format each parameter from signature
        command.help = func_summary
        for p in command.params:
            name = p.name.replace('-', '_')
            # if it is an option
            if isinstance(p, click.Option):
                # use first line of help string from numpy doc
                try:
                    help_str = func_param_docs[name][0]
                except Exception:
                    help_str = cmnd_param_docs[name][0]
                p.help = help_str
            # if it has a default, use the one from the function signature
            if name in func_defaults:
                p.default = func_defaults[name]
            elif name in cmnd_defaults:
                p.default = cmnd_defaults[name]

        return command

    return inner


def ucal_exe_path():
    """
    Get the path to the ucal executable.

    Assumes that the microcalorimetry package is installed in a virtual
    environment, and python.exe is colocated with ucal.exe.

    Returns
    -------
    None.

    """
    bin_dir = Path(sys.executable).parent
    # get a file called ucal
    for file in bin_dir.iterdir():
        if 'ucal' in file.name:
            return file
    raise FileNotFoundError(
        f'program ucal not found in {bin_dir.resolve()}. Is the microcalorimetry package installed in a virtual environment?.'
    )


def ucal_cli(commands: list[str]):
    """
    Execute commands for the ucal CLI in a new terminal.

    The commands are run in a process that is disconnected from the current one
    and must be exited/cleaned up manually. Means I don't have to write the GUI
    to manage multipple processes, keeps it easier.

    Parameters
    ----------
    commands : list[str]
        List of command line arguments to pass into the new terminal that
        is spawned.
    """
    # figure out which new spawn command to use.
    # windows
    ucal = Path(ucal_exe_path())

    if os.name == 'nt':
        start_command = rf'start cmd /K {str(ucal.resolve())}'
    else:
        # if using linux, will need some way to speciy the command for a new
        # terminal (every linux dist will have a different terminal app).
        # Perhaps an environment variable, or a config file
        # in the user directory. Idk, don't want to worry about this right now.
        # IDK if anyone who will use this will even use linux.

        # Making the script runn in a subprocess and piping the console output
        # to the GUI's console is probably the expected approach, but then
        # i have add a way to terminate the script reliably, and worry
        # about someone trying to spin up multiple measurements from the GUI
        # and idk this is just easier for now.
        raise Exception(
            'Only windows currently supported for running measurements on the GUI. Please use the CLI or python API.'
        )

    # execute in new terminal
    command = [start_command] + commands
    command = ' '.join(command)
    print(command)
    os.system(command)


class colors:
    """Colors for logs."""

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def iter_colors(cls):
        attrs = dir(cls)
        for a in attrs:
            if '_' not in a:
                yield a, getattr(cls, a)


class symbols:
    CHECK = '\u2714'
    XBOX = '\u2612'
    BIGX = '\u2a09'

    @classmethod
    def iter_symbols(cls):
        attrs = dir(cls)
        for a in attrs:
            if '_' not in a:
                yield a, getattr(cls, a)


def cstr(*values, color: colors = None):
    """
    Color values

    Parameters
    ----------
    color : colors, optional
        _description_, by default None

    Returns
    -------
    values:
        tuple of strings, so that when passed
        through print they are colored
    """
    cvalues = [v for v in values]
    cvalues[0] = color + str(cvalues[0])
    cvalues[-1] = (cvalues[-1]) + colors.ENDC
    return cvalues


def cprint(*values, color: colors = None, **kwargs):
    """
    Print *values with a color.

    Parameters
    ----------
    color : colors, optional
        Color to print in, None does no color, default prinnt
        statement.
    """
    if color:
        cvalues = cstr(*values, color=color)
        print(*cvalues, **kwargs)
    else:
        print(*values, **kwargs)


class ConsoleManager:
    def __init__(self, fio: io.FileIO = None):
        self.newline_count = 0
        self.set_origin()
        self.fio = fio

    def set_origin(self):
        self.newline_count = 0

    def wipe_to_origin(self):
        for i in range(self.newline_count):
            print('\033[A\033[K', end='')
        self.set_origin()

    def cprint(self, *args, end: str = '\n', sep: str = ' ', log_fio: bool = True):
        msg = sep.join([str(a) for a in args]) + end
        newline_count = msg.count('\n')
        self.newline_count += newline_count
        print(msg, end='')
        # try to log this in the fio, if it fails
        # print the error message to the terminal only
        if self.fio is not None and not self.fio.closed:
            try:
                self.fio.write(msg)
            except Exception as e:
                msg = str(e)
                newline_count = msg.count('\n')
                self.newline_count += newline_count
                print(msg, end='')


if __name__ == '__main__':
    import time

    cm = ConsoleManager()
    print('i should stay')
    cm.set_origin()
    cm.cprint('thing 1', 'thing 2')
    cm.cprint('thing 1', 'thing 3 \n', 'asdas', sep=',', end='\n\n')
    time.sleep(0.5)
    cm.wipe_to_origin()
    cm.cprint('new line')

    my_dict = [
        'hello',
        0,
        {'b': 'there', 'c': 2},
        (0, 'reader'),
        [1, 2, 3, 'person', ['of', 0, {'2': 'interest'}]],
    ]
    for s in recurse_for(my_dict, str):
        print(s)

if __name__ == '__main__':
    path = r'C:\Users\dcg2\Desktop\file.h5'
    grp = 'object_9/0'

    with h5py.File(path, 'r') as f:
        data = _group_saveable.load_object(f[grp], load_big_objects=True)

    save_saveable_objects((data,), path)
