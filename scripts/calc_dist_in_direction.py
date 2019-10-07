import logging

import xarray as xr

from wind_repower_usa.config import INTERIM_DIR
from wind_repower_usa.constants import METER_TO_KM
from wind_repower_usa.geographic_coordinates import calc_location_clusters
from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.wind_direction import calc_dist_in_direction, calc_distance_factors


setup_logging()

turbines = load_turbines()

prevail_wind_direction = xr.open_dataarray(
    INTERIM_DIR / 'wind-direction' / 'prevail_wind_direction.nc')

# this means: distances > 15 * 150m will be ignored, we'll deal only with smaller distances
distance_factor = 15
max_rotor_diameter = 150  # == turbines.t_rd
min_distance_km = distance_factor * max_rotor_diameter * METER_TO_KM

# rotor diameter can be simply increased above, but it could lead to a very slow computation or
# require too much RAM, so it's better to change this manually if necessary
assert turbines.t_rd.max() == max_rotor_diameter, "max rotor diameter changed in turbine database"

logging.info('Cluster turbine locations...')
cluster_per_location, _, _ = calc_location_clusters(turbines, min_distance_km=min_distance_km)

# absolute directions (mathematical orientation, 0rad = east)
logging.info('Calculating distances (absolute direction)...')
distances = calc_dist_in_direction(cluster_per_location,
                                   0 * prevail_wind_direction,
                                   bin_size_deg=15)
distances.attrs['min_distance_km'] = min_distance_km
distances.to_netcdf(INTERIM_DIR / 'distances_in_direction' / 'distances-absolute.nc')


# relative to prevailing wind direction (0deg = prevailing wind direction)
logging.info('Calculating distances (direction relative to prevailing wind)...')
distances = calc_dist_in_direction(cluster_per_location,
                                   prevail_wind_direction,
                                   bin_size_deg=15)
distances.attrs['min_distance_km'] = min_distance_km
distances.to_netcdf(INTERIM_DIR / 'distances_in_direction' / 'distances-relative.nc')

q = 0.05
distance_factors = calc_distance_factors(turbines, distances).quantile(q, dim='turbines')
distance_factors.attrs['min_distance_km'] = min_distance_km
distance_factors.attrs['quantile'] = q
distance_factors.to_netcdf(INTERIM_DIR / 'distances_in_direction' / 'distance_factors.nc')

# TODO should have the parameters in file name or better not? or at least in attr of netcdf?

logging.info('Done!')
