import logging
from itertools import product
from multiprocessing import Pool

import xarray as xr

from wind_repower_usa.config import DISTANCE_FACTORS, INTERIM_DIR
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.turbine_models import new_turbine_models
from wind_repower_usa.optimization import calc_optimal_locations


def calc_optimal_locations_worker(params):
    turbine_model, distance_factor, prevail_wind_direction = params
    logging.info("Started worker for %s, distance_factor=%s",
                 turbine_model.file_name, distance_factor)

    power_generation = xr.open_dataarray(
        INTERIM_DIR / 'simulated_energy_per_location' /
        f'simulated_energy_{turbine_model.file_name}_gwh.nc')

    # TODO this could be already in the *.nc file on generation
    power_generation = power_generation.expand_dims(dim='turbine_model')
    power_generation = power_generation.assign_coords(turbine_model=[turbine_model.file_name])

    if not isinstance(distance_factor, xr.DataArray):
        # this is the old fashion constant distance factor
        optimal_locations = calc_optimal_locations(
            power_generation=power_generation,
            turbine_models=[turbine_model],
            distance_factor=distance_factor
        )
        optimal_locations.attrs['distance_factor'] = distance_factor
        df_filename = f'_{distance_factor}'
    else:
        # this is the distance dependent distance factors
        optimal_locations = calc_optimal_locations(
            power_generation=power_generation,
            turbine_models=[turbine_model],
            distance_factors=distance_factor,
            prevail_wind_direction=prevail_wind_direction,
        )
        df_filename = ''

    optimal_locations.attrs['turbine_model'] = turbine_model.file_name

    optimal_locations.to_netcdf(
        INTERIM_DIR / 'optimal_locations' /
        f'optimal_locations_{turbine_model.file_name}{df_filename}.nc')


def main():
    prevail_wind_direction = xr.open_dataarray(
        INTERIM_DIR / 'wind-direction' / 'prevail_wind_direction.nc')

    distance_factors = xr.open_dataarray(INTERIM_DIR / 'distances_in_direction' /
                                         'distance_factors.nc')

    # TODO the old method wiht constant distance_factors is already quite obsolete
    params = product(new_turbine_models(),
                     DISTANCE_FACTORS + (distance_factors,),
                     [prevail_wind_direction])

    # NUM_PROCESSES=1 because need more RAM, 24GB is not enough for 2 processes
    with Pool(processes=1) as pool:
        pool.map(calc_optimal_locations_worker, params)


if __name__ == '__main__':
    setup_logging()
    main()
