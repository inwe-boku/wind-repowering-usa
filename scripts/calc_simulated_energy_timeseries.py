import os.path as op

from wind_repower_usa.config import YEARS
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.calculations import calc_simulated_energy_years
from wind_repower_usa.turbine_models import ge15_77

setup_logging()

turbine_model = ge15_77

simulated_energy_gwh = calc_simulated_energy_years(YEARS, power_curve=turbine_model.power_curve)

simulated_energy_gwh.to_netcdf(
    op.join('data', 'interim', 'simulated_energy_timeseries',
            f'simulated_energy_timeseries_{turbine_model.file_name}_gwh.nc'))
