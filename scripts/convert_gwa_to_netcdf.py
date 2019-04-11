import logging
import os
import os.path as op

import xarray as xr
from dask.diagnostics import ProgressBar

from wind_repower_usa.logging_config import setup_logging


setup_logging()

dir_names = ('wind_power_density_usa_100m',
             'wind_power_density_usa_200m',
             'wind_power_density_usa_50m',
             'wind_speed_usa_100m',
             'wind_speed_usa_200m',
             'wind_speed_usa_50m')


for dir_name in dir_names:
    logging.info("Starting with %s...", dir_name)
    data = xr.open_rasterio(op.join('data/external/global_wind_atlas_usa', dir_name, '259.tif'),
                            chunks=int(4e3))
    fname = op.join('data/interim/global_wind_atlas_usa', dir_name, f'gwa-usa-{dir_name}.nc')

    os.makedirs(op.dirname(fname), exist_ok=True)

    with ProgressBar():
        data.to_netcdf(fname)
