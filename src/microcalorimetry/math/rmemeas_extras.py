# -*- coding: utf-8 -*-
"""
This module contains functions intended to operate directly on RMEMeas objects,
instead of operating on the underlying datasets though propagation.

"""
from rmellipse.uobjects import RMEMeas
import numpy as np
import xarray as xr

__all__ = ['categorize_by']

def categorize_by(
        meas: RMEMeas,
        category: str,
        combine_uncategorized: bool = True,
        uncategorized_name: str = 'uncategorized',
        deg: bool = False,
        rad: bool = False,
        verbose: bool = False) -> 'RMEMeas':
        """
        Group linear uncertainty mechanisms into categories.

        Mechanisms sharing a common category will be combined quadratically
        into a single mechanism. For example, if the 'Type' category is
        selected, all the mechanisms with the 'Type' = 'A' designation in
        the covcats attribute will be quadratically combined.

        This function does NOT preserve the DOF's or the category information
        of the uncertainty mechanisms after categorization, and is recommended
        to be used for debugging and presentation purposes only. It is NOT
        recommended for categorized RMEMeas objects to be saved.


        Parameters
        ----------
        meas: RMEMeas
            Measurement to group uncertainty mechanisms.
        category : str
            String identifying the category of uncertainty mechanism.
        combine_uncategorized: bool,
            If True, mechanisms that do not contain 'category' will all be
            combined into a single mechanism called uncategorized_name.
        uncategorized_name:str,
            String to call all uncategorized mechanisms IF they are being
            combined. The default is 'uncategorized'.
        deg: bool,
            If true, calculates minimum angle differences for uncertainties
            as degrees.
        rad: bool,
            If true, calculates minimum angle differences for uncertainties
            as radians.
        verbose : bool, optional
            Print information about grouping. The default is False.

        Returns
        -------
        out : RMEMeas
            Measurement.

        """
        assert not all([rad, deg])
        def sum_reexpand(x, k, nominal, deg, rad):
            if deg or rad:
                if deg:
                    const = 180
                else:
                    const = np.pi
                temp = (((x.values - nominal.values) + const) % (2*const) - const)**2
            else:
                temp = (x.values - nominal.values)**2
            temp = np.sum(temp, axis=0)**.5 + nominal.values
            out = nominal.copy()
            out.values = temp
            out = out.assign_coords(umech_id=[k])
            return out

        new = [meas.cov.sel(umech_id=['nominal'])]
        nominal = new[0]

        # do stuff
        covcats = meas.covcats.copy()
        ind = covcats.values == ''
        covcats.values[ind] = 'Uncategorized'

        groupings = np.unique(covcats.loc[:, category])
        groups = {g: list(covcats.umech_id[covcats.loc[:, category] == g].values) for g in groupings}

        try:
            unused_locs = groups.pop('Uncategorized')
            ind = groupings == 'Uncategorized'
            groupings = np.delete(groupings, ind)
        except KeyError:
            unused_locs = []

        for g in groupings:
            new.append(sum_reexpand(meas.cov.sel(umech_id=groups[g]), g, nominal, deg, rad))

        # if combining, pool them into a single mecanism
        if len(unused_locs) > 0 and combine_uncategorized:
            new.append(sum_reexpand(meas.cov.sel(umech_id=unused_locs), uncategorized_name, nominal, deg, rad))

            if verbose:
                print('un mapped umech id being added to ungrouped.')
                print(unused_locs)

        # else, get the unused mechanisms, and copy their covariance categories
        elif len(unused_locs) > 0 and not combine_uncategorized:
            new.append(meas.cov.sel(umech_id=unused_locs))

        # concatenate things
        new_cov = xr.concat(new, 'umech_id')

        if meas.mc is None:
            new_mc = None
        else:
            new_mc = meas.mc.copy()

        out = RMEMeas(name=meas.name, cov=new_cov, mc=new_mc)
        return out

