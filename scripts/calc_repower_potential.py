import logging
from itertools import product
from multiprocessing import Pool

from wind_repower_usa.config import DISTANCE_FACTORS, INTERIM_DIR, NUM_PROCESSES
from wind_repower_usa.load_data import load_optimal_locations, load_simulated_energy_per_location
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.turbine_models import ge15_77, new_turbine_models
from wind_repower_usa.optimization import calc_repower_potential


def calc_optimal_locations_worker(params):
    logging.info("Start calculating repower_potential: %s", params)
    turbine_model_new, distance_factor = params

    turbine_model_old = ge15_77

    power_generation_new = load_simulated_energy_per_location(turbine_model_new)
    power_generation_old = load_simulated_energy_per_location(turbine_model_old,
                                                              capacity_scaling=True)

    optimal_locations = load_optimal_locations(turbine_model_new, distance_factor)
    is_optimal_location = optimal_locations.is_optimal_location
    cluster_per_location = optimal_locations.cluster_per_location

    repower_potential = calc_repower_potential(power_generation_new=power_generation_new,
                                               power_generation_old=power_generation_old,
                                               is_optimal_location=is_optimal_location,
                                               cluster_per_location=cluster_per_location)

    repower_potential.attrs['turbine_model_new'] = turbine_model_new.file_name
    repower_potential.attrs['turbine_model_old'] = turbine_model_old.file_name
    repower_potential.attrs['distance_factor'] = (distance_factor
                                                  if distance_factor is not None
                                                  else 0)

    df_filename = '' if distance_factor is None else f'_{distance_factor}'

    fname = (INTERIM_DIR / 'repower_potential' /
             f'repower_potential_{turbine_model_old.file_name}_'
             f'{turbine_model_new.file_name}{df_filename}.nc')

    repower_potential.to_netcdf(fname)


def main():
    params = product(new_turbine_models(), (None,) + DISTANCE_FACTORS)

    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(calc_optimal_locations_worker, params)


if __name__ == '__main__':
    setup_logging()
    main()
