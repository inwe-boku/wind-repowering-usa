import numpy as np

from wind_repower_usa.load_data import load_turbines, load_wind_speed
from wind_repower_usa.load_data import load_wind_velocity
from wind_repower_usa.calculations import calc_bounding_box_usa, calc_simulated_energy
from wind_repower_usa.calculations import calc_wind_speed_at_turbines


def test_calc_wind_speed_at_turbines():
    turbines = load_turbines()
    year = 2017
    month = 3
    wind_velocity = load_wind_velocity(year, month)
    calc_wind_speed_at_turbines(wind_velocity, turbines)


def test_calc_bounding_box_usa():
    turbines = load_turbines()
    north, west, south, east = calc_bounding_box_usa(turbines)

    # TODO this might be a very specific test, testing also turbines file...
    assert "{}".format(north) == '67.839905'
    assert (west, south, east) == (-172.713074, 16.970871, -64.610001)


def test_calc_simulated_energy():
    turbines = load_turbines()
    wind_speed = load_wind_speed(2016, 7)
    simulated_energy_timeseries_gwh = calc_simulated_energy(wind_speed, turbines)
    np.testing.assert_almost_equal(simulated_energy_timeseries_gwh.isel(time=0).values,
                                   15773.1596734)
