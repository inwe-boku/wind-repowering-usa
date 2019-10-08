import logging
from itertools import product
from multiprocessing import Pool

import xarray as xr

from wind_repower_usa.config import DISTANCE_FACTORS, INTERIM_DIR, NUM_PROCESSES, \
    COMPUTE_CONSTANT_DISTANCE_FACTORS
from wind_repower_usa.load_data import load_optimal_locations, load_simulated_energy_per_location, \
    load_cluster_per_location
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.turbine_models import ge15_77, new_turbine_models
from wind_repower_usa.optimization import calc_repower_potential


def calc_optimal_locations_worker(params):
    logging.info("Start calculating repower_potential: %s", params)
    turbine_model_new, distance_factor = params

    turbine_model_old = ge15_77

    if turbine_model_new == 'mixed':
        turbine_models = new_turbine_models()
    else:
        turbine_models = (turbine_model_new,)

    power_generation_new = xr.concat([load_simulated_energy_per_location(turbine_model_new)
                                      for turbine_model_new in turbine_models],
                                     dim='turbine_model')

    is_optimal_location = xr.concat([load_optimal_locations(turbine_model_new, distance_factor)
                                     for turbine_model_new in turbine_models],
                                    dim='turbine_model')

    power_generation_old = load_simulated_energy_per_location(turbine_model_old,
                                                              capacity_scaling=True)

    cluster_per_location = load_cluster_per_location(distance_factor)

    repower_potential = calc_repower_potential(power_generation_new=power_generation_new,
                                               power_generation_old=power_generation_old,
                                               is_optimal_location=is_optimal_location,
                                               cluster_per_location=cluster_per_location)

    turbine_model_new_fname = (turbine_model_new.file_name
                               if turbine_model_new != 'mixed' else 'mixed')

    repower_potential.attrs['turbine_model_new'] = turbine_model_new_fname
    repower_potential.attrs['turbine_model_old'] = turbine_model_old.file_name
    repower_potential.attrs['distance_factor'] = (distance_factor
                                                  if distance_factor is not None
                                                  else 0)

    df_filename = '' if distance_factor is None else f'_{distance_factor}'

    fname = (INTERIM_DIR / 'repower_potential' /
             f'repower_potential_{turbine_model_old.file_name}_'
             f'{turbine_model_new_fname}{df_filename}.nc')

    repower_potential.to_netcdf(fname)


def main():
    distance_factors = (None,)
    if COMPUTE_CONSTANT_DISTANCE_FACTORS:
        distance_factors += DISTANCE_FACTORS

    params = product(new_turbine_models() + ('mixed',), distance_factors)

    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(calc_optimal_locations_worker, params)


if __name__ == '__main__':
    setup_logging()
    main()
