"""
Created on Feb  2 2026

@author: claudia.rodes
"""


import xarray as xr
import regionmask
import numpy as np
import pandas as pd

from IF_setup import IF_PATH, remove_nas

# 1. Load your data
ds = xr.open_dataset('/scratch/bc3lc/heat-deaths/GCAM_STITCHES_DIPC/results/basd/W5E5v2/CanESM5/Reference/basd/CanESM5_STITCHES_W5E5v2_Reference_tas_global_daily_2015_2050.nc')
tas = ds.tas  # Assuming the variable is named 'tas'

# 2. Load the GCAM mapping CSV
# Skipping the first 6 lines of metadata/comments
mapping_df = pd.read_csv('/scratch/bc3lc/heat-deaths/GCAM_STITCHES_DIPC/input/heat_29012026/iso_GCAM_regID.csv', comment='#')

# 3. Get the Country Mask from Natural Earth
# We use ISO-A3 codes because your CSV uses 'abw', 'afg', etc.
countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
mask = countries.mask(ds.lon, ds.lat)

# 4. Create a Latitude-weighted Mean
# This corrects for the fact that grid cells are smaller at the poles
weights = np.cos(np.deg2rad(ds.lat))
tas_weighted = tas.weighted(weights)

# 5. Group countries into GCAM Regions
gcam_results = {}

# Group the dataframe by the GCAM region name
for region_name, group in mapping_df.groupby('GCAM_region_ID'):
    # Convert ISO codes to lowercase to match typical CSV formatting
    iso_list = group['iso'].str.lower().tolist()
    
    # Find the numeric IDs for these countries in the regionmask database
    # regionmask uses the 'abbreviation' field for ISO-A3 codes
country_ids = []
for iso in iso_list:
    try:
        # map_keys returns the index/number for that abbreviation
        cid = countries.map_keys(iso)
        country_ids.append(cid)
    except (ValueError, KeyError):
        print(f"ISO {iso} not found in this regionmask version.")
        continue
        
    if country_ids:
        # Create a mask for this specific GCAM region
        region_boolean_mask = mask.isin(country_ids)
        
        # Calculate the mean for this region
        gcam_results[region_name] = tas_weighted.mean(dim=('lat', 'lon'), where=region_boolean_mask)

# 6. Combine all regions into one final Dataset
ds_gcam = xr.Dataset(gcam_results)

# Optional: Save to a new NetCDF
# ds_gcam.to_netcdf('tas_by_gcam_region.nc')
