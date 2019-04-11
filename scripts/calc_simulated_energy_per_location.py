import logging

from wind_repower_usa.calculations import calc_simulated_energy
from wind_repower_usa.config import MONTHS, INTERIM_DIR
from wind_repower_usa.load_data import load_wind_speed, load_turbines
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.turbine_models import ge15_77, new_turbine_models

setup_logging()

year = 2017
turbines = load_turbines()
turbine_models = [ge15_77] + list(new_turbine_models())

# simulates the power generation in the current situation for first turbine
capacity_scaling = True

for turbine_model in turbine_models:

    scaling_str = '' if not capacity_scaling else '_capacity_scaled'
    fname = (INTERIM_DIR / 'simulated_energy_per_location' /
             f'simulated_energy_{turbine_model.file_name}{scaling_str}_gwh.nc')

    if fname.exists():
        logging.info("Skipping %s, file already exists", fname)
        continue

    logging.info("Calculating simulated energy for %s", turbine_model.name)

    # power generation per turbine for one year
    wind_speed = load_wind_speed(year, MONTHS)

    # FIXME this is not in gwh, but in gwh/yr or whatever time range wind_speed is!
    simulated_energy_gwh = calc_simulated_energy(wind_speed, turbines,
                                                 power_curve=turbine_model.power_curve,
                                                 sum_along='time',
                                                 capacity_scaling=capacity_scaling,
                                                 only_built_turbines=False)

    simulated_energy_gwh.to_netcdf(fname)

    # done only for first turbine (ge15_77)
    capacity_scaling = False
