import xarray as xr

from wind_repower_usa.config import INTERIM_DIR
from wind_repower_usa.visualize import plot_simulated_generated_energy


def test_plot_simulated_generated_energy():
    simulated_energy_gwh = xr.open_dataarray(
        INTERIM_DIR / 'simulated_energy_timeseries' / 'simulated_energy_timeseries_gwh.nc')
    fig = plot_simulated_generated_energy(simulated_energy_gwh)
    assert fig  # TODO this is a stupid test
