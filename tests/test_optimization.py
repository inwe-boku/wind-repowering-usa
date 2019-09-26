import numpy as np
import xarray as xr

from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.optimization import calc_optimal_locations_cluster, calc_optimal_locations
from wind_repower_usa.turbine_models import e126, Turbine


def locations_to_turbines(locations):
    locations = np.array(locations)
    turbines = xr.Dataset({
            'xlong': ('turbines', locations.T[1]),
            'ylat': ('turbines', locations.T[0]),
        },
        coords={'turbines': np.arange(len(locations))}
    )
    return turbines


def test_find_optimal_locations_cluster_trivial():
    locations = [[0, 0]]
    turbines = locations_to_turbines(locations)

    power_generation = xr.DataArray([[42.]], dims=('turbine_model', 'turbines'))

    is_optimal_location, problem = calc_optimal_locations_cluster(
        turbines=turbines,
        turbine_models=[e126],
        min_distance=10.,
        power_generation=power_generation
    )
    assert problem.status == 'optimal'
    assert problem.solution.opt_val == 42.
    np.testing.assert_equal(is_optimal_location, [[1.]])


def test_find_optimal_locations_cluster():
    locations = [
        [48.2323, 110.223],
        [49.2323, 112.223],
        [48.2423, 110.233],
    ]
    turbines = locations_to_turbines(locations)

    power_generation = xr.DataArray([[42., 23., 41.]], dims=('turbine_model', 'turbines'))

    # 0 <--> 1: 184.05km
    # 1 <--> 2: 182.78km
    # 0 <--> 2: 1.34km
    is_optimal_location, problem = calc_optimal_locations_cluster(
        turbines=turbines,
        turbine_models=[e126],
        min_distance=5.,
        power_generation=power_generation,
    )
    assert problem.status == 'optimal'
    assert problem.solution.opt_val == 65.
    assert is_optimal_location.shape == (1, 3)
    np.testing.assert_equal(is_optimal_location, [[1., 1., 0.]])


def test_find_optimal_locations_only_outliers():
    turbines = load_turbines()
    turbine_model = Turbine(
        name='test turbine',
        file_name='test_turbine',
        power_curve=lambda x: x,
        capacity_mw=1,
        rotor_diameter_m=150.,
        hub_height_m=200.
    )
    power_generation = xr.DataArray(np.ones((1, len(turbines.turbines))), dims=('turbine_model',
                                                                                'turbines'))
    optimal_locations = calc_optimal_locations(
        power_generation=power_generation,
        turbine_models=[turbine_model],
        distance_factor=1e-3,  # needs to be > 0
    )
    # with vanishing distance_factor, we can built at all locations:
    assert np.all(optimal_locations.is_optimal_location == 1.)
    assert optimal_locations.is_optimal_location.shape == (1, len(turbines.turbines))


def test_find_optimal_locations():
    turbines = load_turbines()
    turbine_model = Turbine(
        name='test turbine',
        file_name='test_turbine',
        power_curve=lambda x: x,
        capacity_mw=1,
        rotor_diameter_m=100.,
        hub_height_m=200.
    )
    power_generation = xr.DataArray(np.ones((1, len(turbines.turbines))), dims=('turbine_model',
                                                                                'turbines'))
    optimal_locations = calc_optimal_locations(
        power_generation=power_generation,
        turbine_models=[turbine_model],
        distance_factor=2.5e-2,
    )

    # there are 6 (erroneous) locations with distance < 5.5m to the closest location
    idcs_pairs = [[8489, 56587], [13268, 13391], [54214, 54216], [54215, 54217]]

    # Note that there are erroneous turbine locations, which are only filtered in
    # calc_min_distances_cluster(), which is not used here. It might make sense to clean this
    # data in load_turbines() and use a higher distance_factor for this test.

    is_optimal_location = optimal_locations.is_optimal_location

    for idcs in idcs_pairs:
        assert np.sum(is_optimal_location[0, idcs]) == 1.  # either one or the other is optimal

    assert all(np.delete(is_optimal_location[0], np.array(idcs_pairs).flatten()) == 1.)
