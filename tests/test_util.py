import numpy as np
import xarray as xr
from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.util import turbine_locations, quantile


def test_turbine_locations():
    turbines = load_turbines()
    locations = turbine_locations(turbines)
    assert locations.shape == (turbines.sizes['turbines'], 2)


def test_quantile():
    a = xr.DataArray(np.random.rand(33, 43, 23), dims=('a', 'b', 'c'))

    assert a.quantile(0.23) == quantile(a, 0.23)
    np.testing.assert_array_equal(a.quantile(0.23, dim='b'),
                                  quantile(a, 0.23, dim='b'))

    assert quantile(xr.DataArray([1, 2, 3, np.inf, np.inf, np.inf]), 0.5) == np.inf
