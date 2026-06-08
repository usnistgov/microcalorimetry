"""The stuff I want people to interact with when using the template are in here."""

import customtkinter as ctk
import sys
import os.path as path
from pathlib import Path
import matplotlib as mpl
import microcalorimetry._tkquick._gui._filebar as _filebar
import microcalorimetry._tkquick._gui._graphicsframe as _graphicsframe
import microcalorimetry._tkquick._gui._workingmodes as _workingmodes
from importlib.metadata import version
import matplotlib.backends.backend_tkagg

# Change theme to dark because it doesn't hurt my eyes
ctk.set_appearance_mode('dark')


class GUI(ctk.CTk):
    """
    Entry point for defining an running a GUI.
    """

    def __init__(
        self,
        package_name: str,
        stdout_gui: bool = True,
        stderr_gui: bool = True,
        icon_path: str = None,
        plots_toolbar: matplotlib.backends.backend_tkagg.NavigationToolbar2Tk = None,
        right_sidebar: ctk.CTkFrame = None,
        right_sidebar_kwargs: dict = None,
        right_sidebar_grid: dict = None,
    ):
        """
        Parameters
        ----------
        package_name : str
            Name of the python-package your GUI is bundled with.
        stdout_gui : bool, optional
            If true, prints STDOUT to console, by default True
        stderr_gui : bool, optional
            If true, prints STDERR to console, by default True
        icon_path : str, optional
            Optional path to a custom bitmap icon, by default None
        plots_toolbar : matplotlib.backends.backend_tkagg.NavigationToolbar2Tk, optional
            Subclass of a tkinter plot tool bar, can be provided to customize the tool bar of plots
            embedded in the gui.
        right_sidebar : ctk.CTkFrame = None
            Pass a pointer to a frame class to add to the right sidebar, if desired. Stored under
            the right_sidebar attribute.
        right_sidebar_kwargs: dict
            keyword arguments to pass to the right sidebar.
        right_sidebar_grid: dict
            keyword arguments to pass to the right-sidebar grid.
            If not provided (row=1, column=2, padx=10, pady=(10, 10), sticky='nswe')
        """
        super().__init__()
        version_num = version(package_name)
        self.package_name = package_name
        self.version_num = version_num
        # Set backend so plots can be embedded in GUI manually
        mpl.use('Agg')
        # set default faunts
        self.title_name = None
        self.update_header()
        # sets the minimum and starting window geometry
        self.minsize(1000, 500)
        self.geometry('1000x500')
        self.plots_toolbar = plots_toolbar
        # Sets the icon, should  be a bitmap
        if icon_path is not None:
            self.iconbitmap(path.abspath(path.join(path.dirname(__file__), icon_path)))

        # sets the file bar at the top you can use to load/save the state of the gui
        # it only saves the form fields.
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.fileframe = _filebar.FileFrame(master=self)
        self.fileframe.grid(
            row=0, column=0, padx=10, pady=(10, 0), sticky='new', columnspan=3
        )

        # set up working tabs
        self.workingtabs = _workingmodes.WorkingTabs(master=self)
        self.workingtabs.grid(row=1, column=0, padx=10, pady=10, sticky='nswe')
        self.workingtabs.grid_columnconfigure(0, weight=1)

        # set up a graphics
        self.graphicstabs = _graphicsframe.GraphicsTabs(master=self)

        # add a right sidebar if supplied(file navigator, or whatever)
        if right_sidebar:
            if right_sidebar_kwargs is None:
                right_sidebar_kwargs = {}
            self.right_sidebar = right_sidebar(master=self, **right_sidebar_kwargs)

            if right_sidebar_grid is None:
                self.right_sidebar.grid(
                    row=1, column=2, padx=10, pady=(10, 10), sticky='nswe'
                )
            else:
                self.right_sidebar.grid(**right_sidebar_grid)

            self.graphicstabs.grid(
                row=1, column=1, padx=0, pady=(10, 10), sticky='nswe'
            )
        else:
            self.right_sidebar = None
            self.graphicstabs.grid(
                row=1,
                column=1,
                padx=(0, 10),
                pady=(10, 10),
                sticky='nswe',
                columnspan=2,
            )

        if stdout_gui:
            sys.stdout = self.graphicstabs.console
        if stderr_gui:
            sys.stderr = self.graphicstabs.console

    def update_header(self):
        self.title_name = self.package_name + self.version_num
        self.title(self.title_name)

    def add_function_tab(
        self, name: str, functions: dict[callable], output_group_saveable: list[str]
    ):
        """
        Add a tab for a group of functions

        Parameters
        ----------
        name : str
            What to name the tab.
        functions: dict[Callable]
            Name of attributes under group to make functions for.
        output_group_saveable : list[str]
        """
        self.workingtabs.add_module_tab(
            name, functions, output_groupsaveable=output_group_saveable
        )
