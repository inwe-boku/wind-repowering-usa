import logging
from itertools import product
from multiprocessing import Pool

import xarray as xr

from wind_repower_usa.config import DISTANCE_FACTORS, INTERIM_DIR
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.turbine_models import new_turbine_models
from wind_repower_usa.optimization import calc_optimal_locations


def calc_optimal_locations_worker(params):
    turbine_model, distance_factor = params

    logging.info("Started worker for %s, distance_factor=%s",
                 turbine_model.file_name, distance_factor)

    power_generation = xr.open_dataarray(
        INTERIM_DIR / 'simulated_energy_per_location' /
        f'simulated_energy_{turbine_model.file_name}_gwh.nc')

    optimal_locations = calc_optimal_locations(
        power_generation=power_generation,
        rotor_diameter_m=turbine_model.rotor_diameter_m,
        distance_factor=distance_factor
    )

    optimal_locations.attrs['turbine_model'] = turbine_model.file_name
    optimal_locations.attrs['distance_factor'] = distance_factor

    # TODO add parameters to *.nc file (not only to filename)
    optimal_locations.to_netcdf(
        INTERIM_DIR / 'optimal_locations' /
        f'optimal_locations_{turbine_model.file_name}_{distance_factor}.nc')


def main():
    params = product(new_turbine_models(), DISTANCE_FACTORS)

    # NUM_PROCESSES=1 because need more RAM, 24GB is not enough for 2 processes
    with Pool(processes=1) as pool:
        pool.map(calc_optimal_locations_worker, params)


if __name__ == '__main__':
    setup_logging()
    main()
