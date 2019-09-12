import numpy as np
import xarray as xr

from wind_repower_usa.config import INTERIM_DIR
from wind_repower_usa.load_data import load_turbines, load_optimal_locations
from wind_repower_usa.turbine_models import e138ep3
from wind_repower_usa.wind_direction import calc_dist_in_direction


turbines = load_turbines()

prevail_wind_direction = xr.open_dataarray(
    INTERIM_DIR / 'wind-direction' / 'prevail_wind_direction.nc')

optimal_locations = load_optimal_locations(e138ep3, 4)

cluster_per_location = optimal_locations.cluster_per_location

distances = calc_dist_in_direction(cluster_per_location,
                                   prevail_wind_direction,
                                   bin_size_deg=15)

# FIXME should have some parameters in file name, right?

distances.to_netcdf(INTERIM_DIR / 'wind-direction' / 'distances.nc')
