# -*- coding: utf-8 -*-
"""
Defines array structures used in this library.
"""

from rmellipse import ArraySchema, AnnotatedArray, CoordinateSchema, RMEMeas
from pydantic import BaseModel

frequency_coord = CoordinateSchema(dtype=float, units='GHz')


def as_annotated(
    meas: RMEMeas, Annotation: type[AnnotatedArray], validate: bool = True
) -> RMEMeas[[AnnotatedArray]]:
    """Cast the underlying DataArray of an RMEMeas as an annotated array."""
    new_cov = meas.cov
    if new_cov is not None:
        new_cov = Annotation(new_cov)
        if validate:
            new_cov.validate()

    new_mc = meas.mc
    if new_mc is not None:
        new_mc = Annotation(new_mc)
        if validate:
            new_mc.validate()

    return RMEMeas(
        name=meas.name,
        cov=new_cov,
        mc=new_mc,
        covdofs=meas.covdofs.copy(),
        covcats=meas.covcats.copy(),
    )


class RFSweep(AnnotatedArray):
    """Series of values swept by frequency"""

    schema = ArraySchema(
        shape=(..., 'N'),
        dtype=float,
        dims=(..., 'frequency'),
        coords=dict(
            frequency=frequency_coord,
        ),
    )


class DCSweep(AnnotatedArray):
    """
    Series of values swept by DC


    """

    schema = ArraySchema(
        shape=(..., 'N'),
        dtype=float,
        dims=(..., 'source_setting'),
        units='as',
        coords=dict(
            source_setting=CoordinateSchema(dtype=float),
        ),
    )


class Eta(AnnotatedArray):
    """
    Effect efficiency of a power sensor.

    'eta' is a singular dimension.
    """

    schema = ArraySchema(
        shape=(..., 'N', 1),
        dtype=float,
        dims=(..., 'frequency', 'eta'),
        coords=dict(frequency=frequency_coord, eta={'dtype': int, 'values': [0]}),
    )


class S11(AnnotatedArray):
    """
    Complex reflection coefficient.

    s is a singular dimension.

    """

    schema = ArraySchema(
        shape=(..., 'N', 1),
        dtype=complex,
        dims=(..., 'frequency', 's'),
        coords=dict(frequency=frequency_coord, s={'dtype': str, 'values': ['S11']}),
    )


class GC(AnnotatedArray):
    """
    Correction factor.

    The gc dimension describes the index correction term.
    """

    schema = ArraySchema(
        shape=(..., 'N', 'M'),
        dtype=float,
        dims=(..., 'frequency', 'gc'),
        coords=dict(frequency=frequency_coord, gc={'dtype': int}),
    )


class KDCFitMeta(BaseModel):
    """MetdataRequired for temperature independent thermoelectric sensitivities."""

    """If True, means fit is in W/V, otherwise in V/W."""
    p_of_e: bool


class KDCTempDep(AnnotatedArray):
    "Temperature independent polynomial fit coefficients."

    schema = ArraySchema(
        shape=(..., 'N'),
        dims=(..., 'deg'),
        dtype=float,
        coords=dict(deg=CoordinateSchema(dtype=int)),
        attrs=KDCFitMeta,
    )


class KDCTemInd(AnnotatedArray):
    "Temperature dependent polynomial fit coefficients."

    schema = ArraySchema(
        shape=(..., 10),
        dims=(..., 'col'),
        dtype=float,
        coords=dict(
            col=CoordinateSchema(
                values=[
                    'ap',
                    'bp',
                    'at',
                    'bt',
                    'd',
                    'c',
                    'MeanT',
                    'MeanV',
                    'STDT',
                    'STDV',
                ],
                dtype=str,
            )
        ),
    )
