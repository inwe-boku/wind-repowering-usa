import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN

from wind_repower_usa.constants import EARTH_RADIUS_KM

# TODO rename to km
from wind_repower_usa.load_data import load_cluster_per_location
from wind_repower_usa.util import turbine_locations


def geolocation_distances(locations):
    """Calculate the pairwise distances for geo locations given in lat/long.

    Parameters
    ----------
    locations : np.ndarray
        with shape (N, 2) - in lat/long

    Returns
    -------
    distance matrix in km of shape (N, N) (symmetric, 0. entries in diagonal)

    """
    # FIXME do we need to take care about different coordinate systems or so?
    # FIXME this is not very heavily tested, not sure about correctness, numerical stability etc
    # TODO performance can be improved at least by factor2, ATM it calculates the full (symmetric)
    #  matrix for each element

    # TODO use sklearn instead? seems to support haversine since DBSCAN can do it

    # FIXME should we use something else instead of Haversine?
    #  --> https://en.wikipedia.org/wiki/Vincenty%27s_formulae

    locations_rad = np.radians(locations)
    latitudes, longitudes = locations_rad.T

    latitudes1, latitudes2 = np.meshgrid(latitudes, latitudes)
    longitudes1, longitudes2 = np.meshgrid(longitudes, longitudes)

    a = (np.sin((latitudes2 - latitudes1)/2)**2 + np.cos(latitudes1) *
         np.cos(latitudes2) * np.sin((longitudes2 - longitudes1)/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def calc_min_distances_cluster(locations, n_closest=1):
    """Calculate distances to closest turbine. Meant to be run for one cluster, not for all
    turbines.

    Parameters
    ----------
    locations : shape (N, 2)
    n_closest : int

    """
    MIN_DISTANCE_KM = 5 * 1e-3  # needed to filter out obviously wrong data, like 70cm distances

    distances = geolocation_distances(locations)

    distances = np.where(distances > MIN_DISTANCE_KM, distances, np.inf)
    if n_closest == 1:
        return np.min(distances, axis=0)
    else:
        num_locations = locations.shape[0]
        distances_sorted = np.full((num_locations, n_closest), np.nan)
        cols = min(num_locations, n_closest)
        distances_sorted[:, :cols] = np.sort(distances, axis=1)[:, :n_closest]
        return distances_sorted


def calc_min_distances(locations, cluster_per_location=None, n_closest=1):
    """Calculate distances to closest turbine. Uses clustering to speed up calculation, assuming
    that only distances are relevant which are lower than minimum distances between clusters.

    This method could be also implemented using a spatial index, but it seems to be fast enough
    for the purposes.

    Parameters
    ----------
    locations : shape (N, 2)
    cluster_per_location :
    n_closest : int
        calculate n_closest turbines instead of the min one

    Returns
    -------
    min_distances : np.ndarray of shape (N,) or (N,n_closest) for n_closest>1
        distance in km

    """
    if cluster_per_location is None:
        # it shouldn't matter to much which clustering is used, because we are not interested in
        # min_distances larger than the distance between clusters.
        cluster_per_location = load_cluster_per_location(4)

    clusters = np.unique(cluster_per_location)
    if n_closest == 1:
        closest_location_distances = np.zeros(len(locations))
    else:
        closest_location_distances = np.zeros((len(locations), n_closest))

    for cluster in clusters:
        idcs = cluster == cluster_per_location
        closest_location_distances[idcs] = calc_min_distances_cluster(locations[idcs], n_closest)

    if n_closest == 1:
        return xr.DataArray(closest_location_distances, dims='turbines')
    else:
        return xr.DataArray(closest_location_distances, dims=('turbines', 'n_closest'))


def calc_location_clusters(turbines, min_distance_km=0.5):
    """Calculate a partitioning of locations given in lang/long into clusters using the DBSCAN
    algorithm.

    Runtime: about 10-15 seconds for all turbines.

    Parameters
    ----------
    turbines : xr.DataSet
        as returned by load_turbines()
    min_distance_km : float

    Returns
    -------
    cluster_per_location : xr.DataArray (dims: turbines)
        for each location location the cluster index, -1 for outliers, see
        ``sklearn.cluster.DBSCAN``
    clusters : np.ndarray of shape (M,)
        M is the number of clusters
    cluster_sizes : np.ndarray of shape (M,)
        the size for each cluster

    References
    ----------
    https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size

    """
    locations = turbine_locations(turbines)

    # Parameters for haversine formula
    kms_per_radian = EARTH_RADIUS_KM
    epsilon = min_distance_km / kms_per_radian

    clustering = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree',
                        metric='haversine').fit(np.radians(locations))

    cluster_per_location = clustering.labels_
    clusters, cluster_sizes = np.unique(cluster_per_location, return_counts=True)

    cluster_per_location = xr.DataArray(cluster_per_location, dims='turbines',
                                        coords={'turbines': turbines.turbines},
                                        name='cluster_per_location')  # TODO rename to cluster?

    return cluster_per_location, clusters, cluster_sizes
