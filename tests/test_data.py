"""Tests on input, interim and output data. Mostly very naive tests, just to avoid the most stupid
mistakes."""

import numpy as np

from wind_repower_usa.load_data import load_optimal_locations, load_simulated_energy_per_location
from wind_repower_usa.load_data import load_repower_potential
from wind_repower_usa.turbine_models import e138ep3, ge15_77


def test_optimal_locations():
    optimal_locations = load_optimal_locations(turbine_model=e138ep3, distance_factor=4)
    is_optimal_location = optimal_locations.is_optimal_location.sum(dim='turbine_model')
    cluster_per_location = optimal_locations.cluster_per_location

    num_turbines_per_cluster = is_optimal_location.groupby(cluster_per_location).sum()

    assert all(num_turbines_per_cluster > 0), "each cluster should have at least one new turbine"

    # TODO needs also distance data to closest neighbor:
    # no two optimal locations shall be closer than allowed
    # a non-optimal locations should have a neighbor which is closer than allowed and optimal

    # these values are not really confirmed, but helps against regressions
    assert len(np.unique(cluster_per_location)) == 8964
    assert is_optimal_location.sum() == 28161
    np.testing.assert_array_equal(is_optimal_location[[23, 45, 222, 33438]],
                                  [0, 0, 0, 1])
    np.testing.assert_array_equal(cluster_per_location[[23, 45, 222, 38234]],
                                  [0, 0, 8, 6163])


def test_repowering_potential():
    repower_potential = load_repower_potential(turbine_model_new=e138ep3, distance_factor=4)
    np.testing.assert_allclose(repower_potential.power_generation.sel(num_new_turbines=234),
                               266776.907325)
    assert repower_potential.num_turbines.sel(num_new_turbines=2034) == 58223


def test_repowering_increases_output():
    # not impossible but very unexpected result
    # TODO for all models/factors?
    turbine_model_new = e138ep3
    turbine_model_old = ge15_77

    power_generation_new = load_simulated_energy_per_location(turbine_model_new)
    power_generation_old = load_simulated_energy_per_location(turbine_model_old,
                                                              capacity_scaling=True)
    optimal_locations = load_optimal_locations(turbine_model=e138ep3, distance_factor=4)
    is_optimal_location = optimal_locations.is_optimal_location.sum(dim='turbine_model')
    assert np.sum(is_optimal_location * power_generation_new) > np.sum(power_generation_old)


# TODO test all data for NaNs if they shouldn't contain Nans!
