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
    import microcalorimetry.measurements as measurements
    import microcalorimetry.analysis as anl
    import microcalorimetry.export as export
    from pathlib import Path

    app = GUI(
        'microcalorimetry',
        stdout_gui=no_console_stdout,
        stderr_gui=no_console_stdout,
        right_sidebar=HDF5viewer,
        right_sidebar_kwargs=dict(width=350),
        icon_path=Path(__file__).parent / 'graphics/icon.ico',
    )
    # find installed python packages matching aname pattern lik ('-microcalorimetry')

    # OR look for python scripts you defines some how

    # look for {package_name}.GUI_PLUGIN

    # append as function tab

    app.add_function_tab(
        'measurements.',
        # view function is just parse but with out the ability to save.
        functions={
            'view': measurements.view,
            # 'dcsweep.parse_v0': dcsweep.parse_v0,
            'dcsweep.run': dcsweep._main.run_gui,
            'dcsweep.parse': dcsweep.parse_v1,
            'rfsweep.run': rfsweep._main.run_gui,
            'rfsweep.parse': rfsweep.parse,
            'rfsweep.make_settled_runlist': rfsweep.generate_settled_runlist,
            'rfsweep.make_runlist_from_loss': rfsweep.runlist_from_loss,
            'rfsweep.reduce_initial_power': rfsweep.reduce_initial_power,
            'rfsweep.reorder_runlist': rfsweep.reorder_runlist,
            'rfsweep.review_runlist': rfsweep.review_runlist,
        },
        output_group_saveable=['dcsweep.parse', 'rfsweep.parse'],
    )

    app.add_function_tab(
        'analysis.',
        functions={
            'make_eta_repeatability_model': anl.make_eta_repeatability_model,
            'fit_thermoelectric': anl.fit_thermoelectric,
            'compression_check': anl.compression_check,
            'make_eta': anl.make_eta,
            'dc_lead_correction': anl.dc_lead_correction,
            'apply_uncertainty_model': anl.apply_uncertainty_model,
            'review_eta': anl.review_eta,
            'review_correction': anl.review_correction_factor,
        },
        output_group_saveable=[
            'make_eta_repeatability_model',
            'fit_thermoelectric',
            'compression_check',
            'make_eta',
            'dc_lead_correction',
            'apply_uncertainty_model',
        ],
    )

    app.add_function_tab(
        'export.',
        functions={
            'as_doteff': export.as_doteff,
        },
        output_group_saveable=[],
    )

    app.mainloop()
    pass


# %% Analysis subgroups
# leaving this out for now, might come back to it later.
# @_main.group(
#     name='anl',
#     cls=LazyGroup,
#     lazy_subcommands={
#         'make-k-coeffs': 'microcalorimetry.analysis._sensitivity._cli_make_k_coeffs'
#     },
#     help='Commands for performing analysis.',
# )
# def _anl():
#     pass


# %% DCSweep subgroup
@_main.group(
    name='dcsweep',
    cls=LazyGroup,
    lazy_subcommands={
        'run': 'microcalorimetry.measurements.dcsweep._main._run_cli',
        # 'parse': 'microcalorimetry.measurements.dcsweep._main._parse_cli',
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
