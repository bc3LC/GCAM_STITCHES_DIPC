# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:02:30 2025

@author: theo.rouhette
"""

import numpy as np 
import xarray as xr
import pandas as pd
import os

# COMMON PATHS 
# IF_PATH = "C:/GCAM/Theo/IAM-FIRE"
IF_PATH = "/scratch/bc3lc/heat-deaths/GCAM_STITCHES_DIPC"

# HPC CONDA ENVIRONMENT
conda_env = "heat2"

# EXPERIMENT NAME 
experiment_name = "heat_ineq"

# Function to calculate grid cell area in km²
def grid_cell_area(lat, dlon=0.5, dlat=0.5):
    """Calculate grid cell area given latitude, assuming a 0.5° x 0.5° grid."""
    R = 6371  # Earth radius in km
    lat_rad = np.radians(lat)
    
    # Width of cell (km) varies with latitude
    cell_width = (np.pi / 180) * R * np.cos(lat_rad) * dlon  
    # Height of cell (km)
    cell_height = (np.pi / 180) * R * dlat  
    
    grid_area = cell_width * cell_height 
    
    return grid_area  # Area in km²

def remove_nas(x):
    return x[~pd.isnull(x)]

def generate_scaled_coordinates_helper(coord_min: float,
                                       coord_max: float,
                                       res: float,
                                       ascending: bool = True,
                                       decimals: int = 3) -> np.array:
    """Generate a list of evenly-spaced coordinate pairs for the output grid based on lat, lon values.

    :param coord_min:                   Minimum coordinate in range.
    :type coord_min:                    float

    :param coord_max:                   Maximum coordinate in range.
    :type coord_max:                    float

    :param ascending:                   Ascend coordinate values if True; descend if False
    :type ascending:                    bool

    :param decimals:                    Number of desired decimals to round to.
    :type decimals:                     int

    :returns:                           Array of coordinate values.

    """

    # distance between centroid and edge of grid cell
    center_spacing = res / 2

    if ascending:
        return np.arange(coord_min + center_spacing, coord_max, res).round(decimals)

    else:
        return np.arange(coord_max - center_spacing, coord_min, -res).round(decimals)