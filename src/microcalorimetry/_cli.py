import click
from microcalorimetry._helpers._intf_tools import LazyGroup


@click.group(name='ucal')
def _main():
    pass


# %% Analysis subgroups
@_main.command(name='gui')
@click.option(
    '--no-console-stdout',
    is_flag=True,
    default=False,
    help='Sets the stdout to the GUIs internal console (buggy, a terminal console is nicer).',
)
def _gui(no_console_stdout: bool = False):
    # put import statements here so they are delay until run time
    from microcalorimetry._tkquick import GUI
    from microcalorimetry._tkquick._gui._graphicsframe import HDF5viewer
    import sys
    import microcalorimetry.measurements.dcsweep as dcsweep
    import microcalorimetry.measurements.rfsweep as rfsweep
    import microcalorimetry.analysis as anl
    import microcalorimetry.export as export

    app = GUI(
        'microcalorimetry',
        stdout_gui=no_console_stdout,
        stderr_gui=no_console_stdout,
        right_sidebar=HDF5viewer,
        right_sidebar_kwargs=dict(width=350),
    )

    app.add_function_tab(
        'dcsweep',
        # view function is just parse but with out the ability to save.
        functions={
            'parse': dcsweep.parse,
            'run': dcsweep._main.run_gui,
            'view': dcsweep.parse,
        },
        output_group_saveable=['parse'],
    )

    app.add_function_tab(
        'rfsweep',
        functions={
            'parse': rfsweep.parse,
            'view': rfsweep.view,
            'run': rfsweep._main.run_gui,
            'make settled runlist': rfsweep.generate_settled_runlist,
            'make runlist from loss': rfsweep.runlist_from_loss,
        },
        output_group_saveable=['parse'],
    )

    app.add_function_tab(
        'analysis',
        functions={
            'make eta repeatability model': anl.make_eta_repeatability_model,
            'make k coeffs': anl.make_k_coeffs,
            'make eta': anl.make_eta,
            'dc lead correction': anl.dc_lead_correction,
            'apply_uncertainty_model':anl.apply_uncertainty_model,
            'review eta': anl.review_eta,

        },
        output_group_saveable=[
            'make k coeffs',
            'make eta',
            'make eta hist model',
            'dc lead correction',
        ],
    )

    app.add_function_tab(
        'export',
        functions={
            'as doteff': export.as_doteff,
        },
        output_group_saveable=[],
    )

    app.mainloop()
    pass


# %% Analysis subgroups
@_main.group(
    name='anl',
    cls=LazyGroup,
    lazy_subcommands={
        'make-k-coeffs': 'microcalorimetry.analysis._sensitivity._cli_make_k_coeffs'
    },
    help='Commands for performing analysis.',
)
def _anl():
    pass


# %% DCSweep subgroup
@_main.group(
    name='dcsweep',
    cls=LazyGroup,
    lazy_subcommands={
        'run': 'microcalorimetry.measurements.dcsweep._main._run_cli',
        'parse': 'microcalorimetry.measurements.dcsweep._main._parse_cli',
    },
    help='Commands for interacting with DCSweep measurements.',
)
def _dcsweep():
    pass


@_main.group(
    name='rfsweep',
    cls=LazyGroup,
    lazy_subcommands={
        'run': 'microcalorimetry.measurements.rfsweep._main._run_cli',
    },
    help='Commands for interacting with RFSweep measurements.',
)
def _rfsweep():
    pass
