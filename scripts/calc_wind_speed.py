"""
Generate wind speed data from wind velocity.
"""

import logging
import os.path as op
from dask.diagnostics import ProgressBar

from wind_repower_usa.config import YEARS, MONTHS
from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.load_data import load_wind_velocity
from wind_repower_usa.calculations import calc_wind_speed_at_turbines

from wind_repower_usa.logging_config import setup_logging


def main():
    setup_logging()

    turbines = load_turbines()
    with ProgressBar():
        for year in YEARS:
            for month in MONTHS:
                fname = op.join(
                    'data', 'interim', 'wind_speed_usa_era5',
                    'wind_speed_usa_era5-{}-{:02d}.nc'.format(year, month)
                )

                if op.exists(fname):
                    logging.debug("Skipping %s", fname)
                    continue

                logging.info("Processing %s...", fname)

                wind_velocity = load_wind_velocity(year=year, month=month)
                wind_speed = calc_wind_speed_at_turbines(wind_velocity, turbines)

                wind_speed.to_netcdf(fname)


if __name__ == '__main__':
    main()
