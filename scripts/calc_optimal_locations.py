import logging
from itertools import product
from multiprocessing import Pool

import xarray as xr

from wind_repower_usa.config import DISTANCE_FACTORS, INTERIM_DIR
from wind_repower_usa.load_data import load_prevail_wind_direction
from wind_repower_usa.load_data import load_distance_factors
from wind_repower_usa.load_data import load_cluster_per_location
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
        cluster_per_location = load_cluster_per_location(distance_factor)
        is_optimal_location = calc_optimal_locations(
            power_generation=power_generation,
            turbine_models=[turbine_model],
            cluster_per_location=cluster_per_location,
            distance_factor=distance_factor
        )
        is_optimal_location.attrs['distance_factor'] = distance_factor
        df_filename = f'_{distance_factor}'
    else:
        # this is the distance dependent distance factors
        cluster_per_location = load_cluster_per_location(None)
        is_optimal_location = calc_optimal_locations(
            power_generation=power_generation,
            turbine_models=[turbine_model],
            cluster_per_location=cluster_per_location,
            distance_factors=distance_factor,
            prevail_wind_direction=prevail_wind_direction,
        )
        df_filename = ''

    is_optimal_location.attrs['turbine_model'] = turbine_model.file_name

    is_optimal_location.to_netcdf(
        INTERIM_DIR / 'optimal_locations' /
        f'is_optimal_location_{turbine_model.file_name}{df_filename}.nc')


def main():
    prevail_wind_direction = load_prevail_wind_direction()
    distance_factors = load_distance_factors()

    # TODO the old method wiht constant distance_factors is already quite obsolete
    params = product(new_turbine_models(),
                     DISTANCE_FACTORS + (distance_factors,),
                     [prevail_wind_direction])

    # FIXME there is a deadlock because of logging... :-/
    #  https://codewithoutrules.com/2018/09/04/python-multiprocessing/

    # NUM_PROCESSES=1 because need more RAM, 24GB is not enough for 2 processes
    with Pool(processes=1) as pool:
        pool.map(calc_optimal_locations_worker, params)


if __name__ == '__main__':
    setup_logging()
    main()
