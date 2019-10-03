import logging

from wind_repower_usa.config import INTERIM_DIR, MONTHS
from wind_repower_usa.load_data import load_turbines, load_wind_velocity, load_wind_speed
from wind_repower_usa.wind_direction import calc_wind_rose
from wind_repower_usa.logging_config import setup_logging


setup_logging()

years = range(2010, 2011)

logging.info(f"Calculate prevailing wind direction from years={years}...")

turbines = load_turbines()
wind_velocity = load_wind_velocity(year=years, month=MONTHS)
wind_speed = load_wind_speed(years=2010, months=MONTHS)

wind_rose, prevail_wind_direction, directivity = calc_wind_rose(turbines,
                                                                wind_speed,
                                                                wind_velocity,
                                                                power_curve=None,
                                                                bins=70,
                                                                directivity_width=15)


# TODO should have the parameters in file name or better not?
prevail_wind_direction.to_netcdf(INTERIM_DIR / 'wind-direction' / 'prevail_wind_direction.nc')
wind_rose.to_netcdf(INTERIM_DIR / 'wind-direction' / 'wind_rose.nc')
directivity.to_netcdf(INTERIM_DIR / 'wind-direction' / 'directivity.nc')

logging.info("Done...!")
