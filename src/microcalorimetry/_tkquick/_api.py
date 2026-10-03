"""The stuff I want people to interact with when using the template are in here."""

import customtkinter as ctk
import sys
import os.path as path
from pathlib import Path
import matplotlib as mpl
import microcalorimetry._tkquick._gui._toolbar as _toolbar
import microcalorimetry._tkquick._gui._graphicsframes as _graphicsframes
import microcalorimetry._tkquick._gui._methodframes as _methodframes
import microcalorimetry._tkquick._gui._history as _history
from importlib.metadata import version
import matplotlib.backends.backend_tkagg

# Change theme to dark because it doesn't hurt my eyes
ctk.set_appearance_mode('dark')

# reference to the active application to grab from other modules
_APP: 'GUI' = None


class GUI(ctk.CTk):
    """
    Entry point for defining an running a GUI.
    """

    def __init__(
        self,
        package_name: str,
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
        global _APP
        if _APP is None:
            _APP = self
        else:
            raise Exception('Only one GUI active per process.')
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
        self.fileframe = _toolbar.ToolBarFrame(master=self)
        self.fileframe.grid(
            row=0, column=0, padx=10, pady=(10, 0), sticky='new', columnspan=3
        )

        self.plots_toolbar = plots_toolbar

        # set up tabs for interacting with functions
        self.moduletabs = _methodframes.ModuleTabs(master=self)
        self.moduletabs.grid(row=1, column=0, padx=10, pady=10, sticky='nswe')
        self.moduletabs.grid_columnconfigure(0, weight=1)

        # middle frame other stuff
        self.middle_tabs = ctk.CTkTabview(self)
        self.middle_tabs.grid(row=1, column=1, padx=0, pady=(10, 10), sticky='nswe')
        middle_tab_dict = {
            k: self.middle_tabs.add(k) for k in ['plots', 'history', 'hdf5']
        }
        for name, tab in middle_tab_dict.items():
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        # plot viewer
        self.graphicsframe = _graphicsframes.PlotsFrame(master=middle_tab_dict['plots'])
        self.graphicsframe.grid(padx=0, sticky='nswe')

        # history navigator
        self.hist_frame = _history.HistoryTopLevel(middle_tab_dict['history'])
        self.hist_frame.grid(padx=0, sticky='nswe')

        # hdf5 navigator
        right_sidebar = _graphicsframes.HDF5viewer
        self.right_sidebar = right_sidebar(master=middle_tab_dict['hdf5'])
        self.right_sidebar.grid(padx=0, sticky='nswe')

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
        self.moduletabs.add_module_tab(
            name, functions, output_groupsaveable=output_group_saveable
        )
