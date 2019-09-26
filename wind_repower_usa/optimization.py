import logging

import numpy as np
import cvxpy as cp
import xarray as xr

from wind_repower_usa.config import INTERIM_DIR
from wind_repower_usa.constants import METER_TO_KM
from wind_repower_usa.geographic_coordinates import geolocation_distances, calc_location_clusters
from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.util import turbine_locations
from wind_repower_usa.wind_direction import calc_directions


def _calc_distance_factors(turbines):
    """

    Returns
    -------

    """
    distance_factors_all_turbines = xr.open_dataarray(INTERIM_DIR / 'wind-direction' /
                                                      'distance_factors.nc')

    prevail_wind_direction = xr.open_dataarray(
        INTERIM_DIR / 'wind-direction' / 'prevail_wind_direction.nc')

    distance_factors = distance_factors_all_turbines.quantile(0.05, dim='turbines')

    # for interpolation we need distance factors at least in the interval [-pi, pi], so add here
    # one data point at the beginning and the end by wrapping around (angles are 2*pi periodic)
    end = distance_factors.sel(direction=np.pi, method='nearest', tolerance=0.2)
    end['direction'] -= 2*np.pi
    begin = distance_factors.sel(direction=-np.pi, method='nearest', tolerance=0.2)
    begin['direction'] += 2*np.pi
    distance_factors = xr.concat((begin, distance_factors, end), dim='direction')

    directions = calc_directions(turbines, prevail_wind_direction).fillna(0.)

    distance_factors_turbines = distance_factors.interp(direction=directions).values

    return distance_factors_turbines


def calc_optimal_locations_cluster(turbines, turbine_models, min_distance, power_generation):
    """For a set of locations, this will calculate an optimal subset of locations where turbines
    are to be placed, such that the power generation is maximized and a distance threshold is not
    violated:

    Objective function: maximize sum(power_generation[i], for all turbines i if is optimal location)
    s.t.: distance(i, j) >= min_distance  for i,j where i and j are optimal locations

    This is meant to be run on a small set of locations, e.g. a couple of hundred (or thousands).

    Parameters
    ----------
    turbines : xr.DataSet
        as returned by load_turbines(), but intended to be a subset
    turbine_models : list of turbine_models.Turbine
        used for rotor diameter
    min_distance : float
        in km
    power_generation : xr.DataArray, dims: turbine_model, turbines
        for each turbine (N turbines) an expected power generation, scaling does not matter,
        so it does not matter if it is in GW or GWh/yr or 5*GWh/yr (averaging over 5 years)

    Returns
    -------
    is_optimal_location : np.array of length N
        1 = turbine should be built, 0 = no turbine should be here

    """
    locations = turbine_locations(turbines)
    num_locations = locations.shape[0]
    num_models = len(turbine_models)

    assert len(locations.shape) == 2
    assert locations.shape[1] == 2
    assert power_generation.sizes['turbines'] == num_locations
    assert power_generation.sizes['turbine_model'] == num_models

    pairwise_distances = geolocation_distances(locations)

    # for each location, if True a new turbine should be built, otherwise only decommission old one
    is_optimal_location = cp.Variable((num_models, num_locations), boolean=True)

    rotor_diameter_km = np.array([x.rotor_diameter_m for x in turbine_models]) * METER_TO_KM

    pairwise_distances[np.diag_indices_from(pairwise_distances)] = np.inf

    distance_factors = _calc_distance_factors(turbines)
    # FIXME this just for backward compatibility to check unittests
    distance_factors[:] = min_distance / rotor_diameter_km[0]

    # for a location i, a location j with j != i and a turbine model k at least one of the
    # following must hold:
    #  - k is not built at i    <==> right-hand-side of inequality equals 0 or -1
    #  - nothing is built at j  <==> right-hand-side of inequality equals 0 or -1
    #  - i is far enough away from j for all j != i
    constraints = [pairwise_distances[i, :] / distance_factors[i, :] / rotor_diameter_km[k]
                   >= (is_optimal_location[k, i]
                       + cp.atoms.affine.sum.sum(is_optimal_location, axis=0) - 1)
                   for k in range(num_models) for i in range(num_locations)]

    constraints += [cp.atoms.affine.sum.sum(is_optimal_location, axis=0) <= 1]

    obj = cp.Maximize(cp.atoms.affine.sum.sum(cp.multiply(is_optimal_location,
                                                          power_generation.values)))

    problem = cp.Problem(obj, constraints)

    problem.solve(solver=cp.GUROBI)

    if problem.status != 'optimal':
        raise RuntimeError("Optimization problem could not be"
                           f"solved optimally: {problem.status}")

    return is_optimal_location.value, problem


def calc_optimal_locations(power_generation, turbine_models, distance_factor=3.5):
    """

    Parameters
    ----------
    power_generation : xr.DataArray, dims: turbine_model, turbines
        for each turbine (N turbines) an expected power generation, scaling does not matter,
        so it does not matter if it is in GW or GWh/yr or 5*GWh/yr (averaging over 5 years)
    turbine_models : list of turbine_models.Turbine
        will try to find an optimal configuration of these turbine models (mixed also within one
        cluster), this is mostly used for rotor diameter, list needs to match with dimension
        `turbine_model` in parameter `power_generation`
    distance_factor : float


    Returns
    -------
    optimal_locations : xr.Dataset, dims: turbine_model, turbines (length N)
        variable 'is_optimal_location': 1 = turbine should be built, 0 = no turbine should be here
        variable 'cluster_per_location': see ``calc_location_clusters()``

    """
    # TODO actually only min_distance has to be removed, then it should work! add also unittests!
    assert len(turbine_models) == 1, "multiple turbine models not yet supported"
    rotor_diameter_m = turbine_models[0].rotor_diameter_m
    min_distance = distance_factor * rotor_diameter_m * METER_TO_KM

    turbines = load_turbines()
    locations = turbine_locations(turbines)

    cluster_per_location, clusters, cluster_sizes = calc_location_clusters(locations, min_distance)

    is_optimal_location = np.ones((len(turbine_models), len(cluster_per_location)))

    # clusters[0] should be cluster -1, i.e. outliers which can be always True

    # TODO this should be replaced by looping over groupby() --> speedup by a couple of minutes
    for cluster in clusters[1:]:
        logging.info(f"Optimizing cluster {cluster}...")
        locations_in_cluster = cluster == cluster_per_location
        is_optimal_location_cluster, problem = calc_optimal_locations_cluster(
            turbines=turbines.sel(turbines=locations_in_cluster),
            turbine_models=turbine_models,
            min_distance=min_distance,        # FIXME goes away!
            power_generation=power_generation.sel(turbines=locations_in_cluster)
        )

        is_optimal_location[:, locations_in_cluster] = is_optimal_location_cluster

    optimal_locations = xr.Dataset({'is_optimal_location': (('turbine_model', 'turbines'),
                                                            is_optimal_location),
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
