import logging

import numpy as np
import cvxpy as cp
import xarray as xr

from wind_repower_usa.constants import METER_TO_KM
from wind_repower_usa.geographic_coordinates import geolocation_distances, calc_location_clusters
from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.util import turbine_locations


def calc_optimal_locations_cluster(locations, min_distance, power_generation):
    """For a set of locations, this will calculate an optimal subset of locations where turbines
    are to be placed, such that the power generation is maximized and a distance threshold is not
    violated.

    This is mean to be run on a small set of locations, e.g. a couple of hundred (or thousands).

    Parameters
    ----------
    locations : np.ndarray
        with shape (N, 2) - in lat/long
    min_distance : float
        in km
    power_generation : np.ndarray
        for each turbine (N turbines) an expected power generation, scaling does not matter,
        so it does not matter if it is in GW or GWh/yr or 5*GWh/yr (averaging over 5 years)

    Returns
    -------
    is_optimal_location : np.array of length N
        1 = turbine should be built, 0 = no turbine should be here

    """
    num_locations = locations.shape[0]

    assert len(locations.shape) == 2
    assert locations.shape[1] == 2
    assert power_generation.shape == (num_locations,)

    # for each location, if True a new turbine should be built, otherwise only decommission old one
    is_optimal_location = cp.Variable(num_locations, boolean=True)

    pairwise_distances = geolocation_distances(locations)

    # TODO not sure if it is faster to have this vectorized with a huge matrix or not vectorized
    #  with a smaller list
    lhs = pairwise_distances/min_distance
    upper_triangle = ~np.tri(*pairwise_distances.shape, dtype=np.bool)
    constraints = [lhs[i, j] >= (is_optimal_location[i] + is_optimal_location[j] - 1)
                   for i, j in zip(*np.where((pairwise_distances < min_distance) &
                                             upper_triangle))]

    logging.info("Number of locations: %s", num_locations)
    logging.info("Number of constraints: %s", len(constraints))

    obj = cp.Maximize(power_generation @ is_optimal_location)
    problem = cp.Problem(obj, constraints)

    problem.solve(solver=cp.GUROBI)

    if problem.status != 'optimal':
        raise RuntimeError("Optimization problem could not be"
                           f"solved optimally: {problem.status}")

    return is_optimal_location.value, problem


def calc_optimal_locations(power_generation, rotor_diameter_m, distance_factor=3.5):
    """

    Parameters
    ----------
    power_generation : xr.DataArray of length N
        for each turbine (N turbines) an expected power generation, scaling does not matter,
        so it does not matter if it is in GW or GWh/yr or 5*GWh/yr (averaging over 5 years)
    rotor_diameter_m : float

    distance_factor : float


    Returns
    -------
    optimal_locations : xr.Dataset, dims: turbines (length N)
        variable 'is_optimal_location': 1 = turbine should be built, 0 = no turbine should be here
        variable 'cluster_per_location': see ``calc_location_clusters()``

    """
    min_distance = distance_factor * rotor_diameter_m * METER_TO_KM

    turbines = load_turbines()
    locations = turbine_locations(turbines)

    cluster_per_location, clusters, cluster_sizes = calc_location_clusters(locations, min_distance)

    is_optimal_location = np.ones_like(cluster_per_location)

    # clusters[0] should be cluster -1, i.e. outliers which can be always True

    for cluster in clusters[1:]:
        logging.info(f"Optimizing cluster {cluster}...")
        locations_in_cluster = cluster == cluster_per_location
        is_optimal_location_cluster, problem = calc_optimal_locations_cluster(
            locations=locations[locations_in_cluster, :],
            min_distance=min_distance,
            power_generation=power_generation[locations_in_cluster].values
        )

        is_optimal_location[locations_in_cluster] = is_optimal_location_cluster

    optimal_locations = xr.Dataset({'is_optimal_location': ('turbines', is_optimal_location),
                                    'cluster_per_location': ('turbines', cluster_per_location)})

    return optimal_locations


def calc_repower_potential(power_generation_new, power_generation_old, optimal_locations):
    """Calculate total average power generation and total number of turbines per number of new
    installed turbines.

    'new' refers to the repowered turbine or cluster, 'old' to the currently built turbines.

    Parameters
    ----------
    power_generation_new : xr.DataArray of length N
        for each turbine (N turbines) an expected power generation, scaling does not matter,
        so it does not matter if it is in GW or GWh/yr or 5*GWh/yr (averaging over 5 years)
    power_generation_old : xr.DataArray of length N
        as ``power_generation_new`` but for the currently installed turbines, i.e. with a power
        curve which is currently used and with capacity scaling
    optimal_locations :
        the optimal choice of locations for the turbine model used for power_generation_new,
        see ``calc_optimal_locations()``

    Returns
    -------
    repower_potential : xr.Dataset

    """
    # TODO this function could probably run faster (25s ATM) with a bit more sophisticated code

    is_optimal_location = optimal_locations.is_optimal_location
    cluster_per_location = optimal_locations.cluster_per_location

    # can be negative if distances between locations are very close and not many turbines fit in
    power_gain_per_turbine = power_generation_new * is_optimal_location - power_generation_old

    cluster_per_location = cluster_per_location.copy()  # copy before modify is cheap & safer

    # convert outliers where cluster == -1 to single clusters
    num_outliers = np.sum(cluster_per_location == -1)
    new_cluster_label_start = cluster_per_location.max() + 1
    cluster_per_location[cluster_per_location == -1] = np.arange(
        new_cluster_label_start, new_cluster_label_start + num_outliers)

    cluster_per_location = xr.DataArray(cluster_per_location, dims='turbines', name='cluster')

    # cluster_sizes_old has been already calculated in calc_location_clusters(), but doesn't matter
    _, cluster_sizes_old = np.unique(cluster_per_location, return_counts=True)
    cluster_sizes_new = is_optimal_location.groupby(cluster_per_location).sum()

    # just a naive plausibility test, would be nicer to move this to unit tests
    assert np.sum(cluster_sizes_old < cluster_sizes_new) == 0, ("some clusters have more turbines "
                                                                "after repowering")

    power_gain_per_cluster = power_gain_per_turbine.groupby(cluster_per_location).sum()
    power_per_cluster_mean = power_gain_per_cluster / cluster_sizes_new

    # sort clusters decreasing by average power per new (repowered) turbine in cluster
    cluster_idcs = np.argsort(power_per_cluster_mean)[::-1].values

    # assume the first n clusters in the sorted list are repowered

    power_generation_new_feasable = power_generation_new * is_optimal_location
    power_per_cluster_new = power_generation_new_feasable.groupby(cluster_per_location).sum().values
    power_per_cluster_old = power_generation_old.groupby(cluster_per_location).sum().values

    def reverse_cumsum(a):
        return np.hstack(((a[::-1].cumsum())[::-1][1:], [0]))

    total_power_new = np.cumsum(power_per_cluster_new[cluster_idcs])
    total_power_old = reverse_cumsum(power_per_cluster_old[cluster_idcs])

    power_generation = total_power_new + total_power_old

    number_new_turbines = np.cumsum(cluster_sizes_new.values[cluster_idcs])
    number_old_turbines = reverse_cumsum(cluster_sizes_old[cluster_idcs])

    # TODO num_new_turbines == 0 is missing, could be added to extend the plot
    repower_potential = xr.Dataset({
        'power_generation': ('num_new_turbines', power_generation),
        'num_turbines': ('num_new_turbines', number_new_turbines + number_old_turbines)},
        coords={'num_new_turbines': number_new_turbines}
    )
    return repower_potential
