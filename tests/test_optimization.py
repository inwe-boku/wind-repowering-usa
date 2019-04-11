import numpy as np
import xarray as xr

from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.optimization import calc_optimal_locations_cluster, calc_optimal_locations


def test_find_optimal_locations_cluster_trivial():
    locations = [[0, 0]]
    is_optimal_location, problem = calc_optimal_locations_cluster(
        locations=np.array(locations),
        min_distance=10.,
        power_generation=np.array([42.])
    )
    assert problem.status == 'optimal'
    assert problem.solution.opt_val == 42.
    assert all(is_optimal_location == [1.])


def test_find_optimal_locations_cluster():
    locations = [
        [48.2323, 110.223],
        [49.2323, 112.223],
        [48.2423, 110.233],
    ]
    # 0 <--> 1: 184.05km
    # 1 <--> 2: 182.78km
    # 0 <--> 2: 1.34km
    is_optimal_location, problem = calc_optimal_locations_cluster(
        locations=np.array(locations),
        min_distance=5.,
        power_generation=np.array([42., 23., 41.])
    )
    assert problem.status == 'optimal'
    assert problem.solution.opt_val == 65.
    assert all(is_optimal_location == [1., 1., 0.])


def test_find_optimal_locations_only_outliers():
    turbines = load_turbines()
    optimal_locations = calc_optimal_locations(
        power_generation=np.ones_like(len(turbines.turbines)),
        rotor_diameter_m=150,
        distance_factor=1e-3,  # needs to be > 0
    )
    # with vanishing distance_factor, we can built at all locations:
    assert all(optimal_locations.is_optimal_location == 1.)


def test_find_optimal_locations():
    turbines = load_turbines()
    power_generation = xr.DataArray(np.ones(len(turbines.turbines)))
    optimal_locations = calc_optimal_locations(
        power_generation=power_generation,
        rotor_diameter_m=100,
        distance_factor=2.5e-2,
    )

    # there are 6 (erroneous) locations with distance < 2.5m to the closest location
    idcs_pairs = [[13418, 13258], [54383, 54381], [54385, 54382]]

    is_optimal_location = optimal_locations.is_optimal_location

    for idcs in idcs_pairs:
        assert np.sum(is_optimal_location[idcs]) == 1.  # either one or the other is optimal

    assert all(np.delete(is_optimal_location, np.array(idcs_pairs).flatten()) == 1.)
