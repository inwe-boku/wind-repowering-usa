from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.util import turbine_locations


def test_turbine_locations():
    turbines = load_turbines()
    locations = turbine_locations(turbines)
    assert locations.shape == (turbines.sizes['turbines'], 2)
