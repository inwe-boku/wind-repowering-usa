from wind_repower_usa.config import INTERIM_DIR
from wind_repower_usa.geographic_coordinates import calc_min_distances
from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.util import turbine_locations


setup_logging()

turbines = load_turbines()
locations = turbine_locations(turbines)

min_distances = calc_min_distances(locations)
min_distances.to_netcdf(INTERIM_DIR / 'min_distances' / 'min_distances.nc')
