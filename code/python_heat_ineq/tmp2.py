"""
The `fx_stitch` module is responsible for stitching together climate model outputs.

It creates a continuous time series that can be used for climate analysis and emulation.

"""

import os
from importlib import resources

import numpy as np
import pandas as pd
import xarray as xr

import stitches.fx_data as data
import stitches.fx_pangeo as pangeo
import stitches.fx_util as util


def find_zfiles(rp):
    """
    Determine which CMIP files must be downloaded from Pangeo.
    :param rp: Data frame of the recipes.
    :return: Numpy ndarray of the gs:// files to pull from Pangeo.
    """
    # Figure out which columns contain the string file
    flat_list = rp.filter(regex="file", axis=1).values.flatten()
    unique_list = np.unique(flat_list)
    return unique_list



def find_var_cols(x):
    """
    Determine the variables to be downloaded.
    :param x: pandas DataFrame of the stitches recipe.
    :return: List of variables to be written to the NetCDF files.
    """
    # Parse out the variable name so that we can use it
    # to label the final output.
    set = x.filter(regex="file").columns.tolist()
    out = []
    for text in set:
        new = text.replace("_file", "")
        out.append(new)
    return out



def get_netcdf_values(i, dl, rp, fl, name):
    """
    Extract archive values from a list of downloaded CMIP data.

    :param i: Index of the row in the recipe data frame.
    :param dl: List of xarray datasets containing CMIP files.
    :param rp: DataFrame of the recipe.
    :param fl: List of CMIP file paths.
    :param name: Name of the variable file to process.
    :return: A slice of xarray data (unsure about the technical term).
    """
    file = rp[name][i]
    start_yr = rp["archive_start_yr"][i]
    end_yr = rp["archive_end_yr"][i]
    # Figure out which index level we are on and then get the
    # xarray from the list.
    index = int(np.where(fl == file)[0])
    extracted = dl[index].sortby("time")
    v = name.replace("_file", "")
    # Have to have special time handler
    times = extracted.indexes["time"]
    if type(times) in [
        xr.coding.cftimeindex.CFTimeIndex,
        pd.core.indexes.datetimes.DatetimeIndex,
    ]:
        yrs = extracted.indexes[
            "time"
        ].year  # pull out the year information from the time index
        flags = list(map(lambda x: x in range(start_yr, end_yr + 1), yrs))
        to_keep = times[flags]
    else:
        raise TypeError("Unsupported time type.")
    dat = extracted.sel(time=to_keep)[v].values.copy()
    if (times.freq == "D") | (times.freq == "day"):
        expected_times = pd.date_range(
            start=str(start_yr) + "-01-01", end=str(end_yr) + "-12-31", freq="D"
        )
        if times.calendar == "noleap":
            expected_len = len(
                expected_times[
                    ~((expected_times.month == 2) & (expected_times.day == 29))
                ]
            )
    else:
        expected_len = len(
            pd.date_range(
                start=str(start_yr) + "-01-01", end=str(end_yr) + "-12-31", freq="M"
            )
        )
    assert len(dat) == expected_len, (
        "Not enough data in " + file + "for period " + str(start_yr) + "-" + str(end_yr)
    )
    return dat




def get_var_info(rp, dl, fl, name):
    """
    Extract the CMIP variable attribute information.
    :param rp: Data frame of the recipes.
    :param dl: List of the data files.
    :param fl: List of the data file names.
    :param name: String of the column containing the variable file name from rp.
    :return: Pandas dataframe of the variable meta data.
    """
    util.check_columns(rp, {name})
    file = rp[name][0]
    index = int(np.where(fl == file)[0])
    extracted = dl[index]
    attrs = data.get_ds_meta(extracted)
    if attrs.frequency.values != "mon":
        attrs["calendar"] = extracted.indexes["time"].calendar
    return attrs




def get_atts(rp, dl, fl, name):
    """
    Extract the CMIP variable attribute information.
    :param rp: Data frame of the recipes.
    :param dl: List of the data files.
    :param fl: List of the data file names.
    :param name: String of the column containing the variable file name from rp.
    :return: Dict object containing the CMIP variable information.
    """
    file = rp[name][0]
    index = int(np.where(fl == file)[0])
    extracted = dl[index]
    v = name.replace("_file", "")
    out = extracted[v].attrs.copy()
    return out




def internal_stitch(rp, dl, fl):
    """
    Stitch a single recipe into netCDF outputs.
    :param dl: List of xarray CMIP files.
    :param rp: DataFrame of the recipe.
    :param fl: List of the CMIP file names.
    :return: List of the data arrays for the stitched products of the different variables.
    """
    rp = rp.sort_values(by=["stitching_id", "target_start_yr"]).copy()
    rp.reset_index(drop=True, inplace=True)
    variables = find_var_cols(rp)
    out = []
    # For each of the of the variables stitch the
    # data together.
    for v in variables:
        # Get the information about the variable that is going to be stitched together.
        col = v + "_file"
        var_info = get_var_info(rp, dl, fl, col)
        # For each of time slices extract the data & concatenate together.
        gridded_data = get_netcdf_values(i=0, dl=dl, rp=rp, fl=fl, name=col)
        # Now add the other time slices.
        for i in range(1, len(rp)):
            new_vals = get_netcdf_values(i=i, dl=dl, rp=rp, fl=fl, name=col)
            gridded_data = np.concatenate((gridded_data, new_vals), axis=0)
        # Note that the pd.date_range call need the date/month defined otherwise it will
        # truncate the year from start of first year to start of end year which is not
        # what we want. We want the full final year to be included in the times series.
        start = str(min(rp["target_start_yr"]))
        end = str(max(rp["target_end_yr"]))
        if var_info["frequency"][0].lower() == "mon":
            freq = "M"
        elif var_info["frequency"][0].lower() == "day":
            freq = "D"
        else:
            raise TypeError("Unsupported frequency.")
        times = pd.date_range(start=start + "-01-01", end=end + "-12-31", freq=freq)
        # Again, some ESMs stop in 2099 instead of 2100 - so we just drop the
        # last year of gridded_data when that is the case.
        # TODO this will need something extra/different for daily data; maybe just
        # a simple len(times)==len(gridded_data)-12 : len(times) == len(gridded_data)-(nDaysInYear)
        # with correct parentheses would do it
        if (max(rp["target_end_yr"]) == 2099) & (
            len(times) == (len(gridded_data) - 12)
        ):
            gridded_data = gridded_data[0 : len(times), 0:, 0:].copy()
        if freq == "D":
            if (var_info["calendar"][0].lower() == "noleap") & (freq == "D"):
                times = times[~((times.month == 2) & (times.day == 29))]
        assert len(gridded_data) == len(times), "Problem with the length of time"
        # Extract the lat and lon information that will be used to structure the
        # empty netcdf file. Make sure to copy all of the information including
        # the attributes!
        lat = dl[0].lat.copy()
        lon = dl[0].lon.copy()
        rslt = xr.Dataset(
            {
                v: xr.DataArray(
                    gridded_data,
                    coords=[times, lat, lon],
                    dims=["time", "lat", "lon"],
                    attrs={
                        "units": var_info["units"][0],
                        "variable": var_info["variable"][0],
                        "experiment": var_info["experiment"][0],
                        "ensemble": var_info["ensemble"][0],
                        "model": var_info["model"][0],
                        "stitching_id": rp["stitching_id"].unique()[0],
                    },
                )
            }
        )
        out.append(rslt)
    out_dict = dict(zip(variables, out))
    return out_dict
