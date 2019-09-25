import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from wind_repower_usa.calculations import calc_simulated_energy
from wind_repower_usa.constants import KM_TO_METER
from wind_repower_usa.geographic_coordinates import geolocation_distances
from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.util import turbine_locations, edges_to_center


def calc_wind_rose(turbines, wind_speed, wind_velocity, power_curve=None, bins=70,
                   directivity_width=15):
    """Calculate prevailing wind direction for each turbine location in ``turbines``. A wind rose is
    calculated by the amount of energy produced by wind blowing in a certain wind direction using a
    specific power curve. Note that definition of wind rose differs slightly from usual
    conventions, because this calculates the amplitude using produced energy and not wind speed.

    Parameters
    ----------
    turbines : xr.DataSet
        as returned by load_turbines()
    wind_speed : xr.DataArray
        as returned by load_wind_speed()
    wind_velocity : xr.Dataset
        as downloaded from ERA5
    power_curve : callable
        a function mapping wind speed to power
    bins : int
        bins for histogram of distribution of energy (~wind speed) over direction
    directivity_width : float (in degree)
        see directivity below

    Returns
    -------
    wind_rose : xr.DataArray
        contains ratio of wind energy produced into a certain direction for each location for the
        given power_curve, i.e. integral over dim=direction should be 1.
        dims = turbines, direction
    prevail_wind_direction : np.array
        direction in rad for each turbine location (between -np.pi and np.pi, i.e. 0 is east,
        np.pi/2 is north)
    directivity : np.array
        For each turbine location, directivity is defined as percentage of energy in a certain
        direction with angle ``directivity_width``. This means setting this to 15° means that a
        result of directivity=0.4 that there is a direction which leads to 40% of energy
        production in 15° angle (that is +/- 7.5°). The direction is given by
        prevail_wind_direction.

    """
    # TODO speed up potential of this function:
    #  - choose some wind speed samples only
    #  - calculate only once for entire park

    # interpolation is already done, but only stored as wind speed, u/v components not separately
    wind_velocity_at_turbines = wind_velocity.interp(
        longitude=xr.DataArray(turbines.xlong.values, dims='turbines'),
        latitude=xr.DataArray(turbines.ylat.values, dims='turbines'),
        method='linear')

    # TODO this might be more accurate using Vincenty’s formula, right? Or is wind direction
    #  different from calculating Azimuth? Anyway it should be good enough for our purposes.
    #  see also Azimuth calculator: https://www.cqsrg.org/tools/GCDistance/

    # FIXME compare differences between 100m and 10m
    directions = np.arctan2(wind_velocity_at_turbines.v100,
                            wind_velocity_at_turbines.u100).compute()

    energy = calc_simulated_energy(wind_speed,
                                   turbines,
                                   power_curve=power_curve,
                                   sum_along='',
                                   only_built_turbines=False)

    boxcar_width_angle = np.radians(directivity_width)

    wind_roses_list = []

    directivity = []
    prevail_wind_direction = []

    # would be great to use np.histogramdd() instead of this loop but somehow doesn't seem to work
    for turbine_idx in range(directions.values.shape[1]):
        hist = np.histogram(directions.values.T[turbine_idx, :],
                            weights=energy.values.T[turbine_idx, :],
                            range=(-np.pi, np.pi), bins=bins, density=True)
        values, bin_edges = hist
        wind_roses_list.append(values)

        # TODO calculating this inside loop is bad, should yield same results every iteration
        bin_centers = edges_to_center(bin_edges)

        boxcar_width = int(np.round(boxcar_width_angle / (2 * np.pi) * bins))

        #
        convoluted = uniform_filter(values, boxcar_width, mode='wrap')

        # In case of multiple maxima it might make sense to take the central one or so,
        # but this can only occur if wind speed is equally strong in an interval larger than
        # boxcar_width.
        prevail_wind_direction.append(bin_centers[np.argmax(convoluted)])

        # TODO this value might not really make sense that way, actually one needs a whole
        #  profile for different values of boxcar_width
        directivity.append(np.max(convoluted) * boxcar_width_angle)

    wind_rose = xr.DataArray(wind_roses_list,
                             dims=('turbines', 'direction'),
                             coords={'direction': bin_centers,
                                     'turbines': turbines.turbines})

    prevail_wind_direction_xr = xr.DataArray(prevail_wind_direction, dims='turbines',
                                             coords={'turbines': turbines.turbines})
    directivity = xr.DataArray(directivity, dims='turbines',
                               coords={'turbines': turbines.turbines})

    return wind_rose, prevail_wind_direction_xr, directivity


def calc_directions(turbines, prevail_wind_direction=None):
    """Calculate pairwise directions from each turbine location to each other turbine location.

    Parameters
    ----------
    turbines : xr.DataSet
        as returned by load_turbines()
    prevail_wind_direction : xr.DataArray  (dim = turbines)
        will be used to orientate distances relative to prevailing wind direction,
        pass an xr.DataArray with zeros to get distances per absolute directions (not relative to
        prevailing wind direction)

    Returns
    -------
    xr.DataArray (dims: turbines, targets)
        direction of vector from turbines to target, where targets is a copy of turbines relative to
        prevailing wind direction, NaN in diagonal to avoid subsequent mistakes when calculating
        with diagonal accidentally

    """
    # targets are a copy of turbines: for each turbine locations angle of the vector to each
    # target location will be calculated, sorted into bins of regular angles and then the closest
    # turbine per bin is chosen to assign a distance to turbine per direction.
    targets = turbines.rename({'turbines': 'targets'})

    # pairwise directions from each turbine to each other one - meshgrid magic using xarray, yeah!
    directions = np.arctan2(targets.ylat - turbines.ylat, targets.xlong - turbines.xlong)

    if prevail_wind_direction is not None:
        directions = directions - prevail_wind_direction

    # all angles in mathematical orientation between -pi and pi
    directions = (directions + np.pi) % (2 * np.pi) - np.pi

    # there is no real meaning to calculate the rotation of a vector of length 0...
    directions.values[np.diag_indices_from(directions.values)] = np.nan

    return directions


def calc_dist_in_direction_cluster(turbines, prevail_wind_direction, bin_size_deg=15):
    """Same as calc_dist_in_direction(), but intended for one cluster only. Calculates a squared
    distance matrix (and a squared direction matrix) and therefore RAM usage is O(len(turbines)^2).

    Parameters
    ----------
    turbines : xr.DataSet
        as returned by load_turbines()
    prevail_wind_direction : xr.DataArray  (dim = turbines)
        will be used to orientate distances relative to prevailing wind direction,
        pass an xr.DataArray with zeros to get distances per absolute directions (not relative to
        prevailing wind direction)
    bin_size_deg : float
        size of direction bins in degrees

    Returns
    -------
    xr.DataArray
        dims: turbines, direction
        direction is relative to prevail_wind_direction, i.e. 0° = in prevailing wind direction,
        and otherwise counter-clockwise relative to 0°

    """
    directions = calc_directions(turbines, prevail_wind_direction)

    # directions is actually not used here, because bins and range are provided (except for dtype)
    bin_edges = np.histogram_bin_edges(directions,
                                       bins=360//bin_size_deg,
                                       range=(-np.pi, np.pi))

    num_bins = len(bin_edges) - 1  # Attention, fencepost problem!

    # np.digitize does not return the n-th bin, but the n+1-th bin!
    # This is not a symmetric matrix, directions get flipped by 180° if dims is provided in wrong
    # order, but it is not at all clear how xarray defines the order (probably the order of
    # usage of dims 'targets' and 'turbines' in the arctan2() call above).
    bin_idcs = np.digitize(directions, bin_edges) - 1
    bin_idcs = xr.DataArray(bin_idcs, dims=('targets', 'turbines'),  # targets = closest turbines
                            coords={'turbines': turbines.turbines})

    locations = turbine_locations(turbines)
    distances = geolocation_distances(locations)

    # set distance to itself to INF to avoid zero distance minimums later
    distances[np.diag_indices_from(distances)] = np.inf
    distances = xr.DataArray(distances, dims=('turbines', 'targets'),
                             coords={'turbines': turbines.turbines})

    bin_centers = edges_to_center(bin_edges)
    direction_bins = xr.DataArray(np.arange(num_bins), dims='direction',
                                  coords={'direction': bin_centers})

    return xr.where(bin_idcs == direction_bins, distances, np.inf).min(dim='targets')


def calc_dist_in_direction(cluster_per_location, prevail_wind_direction, turbines=None,
                           bin_size_deg=15):
    """Directions between 0° and 360° will be grouped into bins of size ``bin_size_deg``,
    then for each turbine location the distance to the next turbine is calculated for each
    direction bin. Assumes that distance between clusters is infinite and therefore computation
    can be done for each cluster independently.

    Parameters
    ----------
    cluster_per_location : array_like of int
        cluster index for each turbine
    prevail_wind_direction : xr.DataArray  (dim = turbines)
        will be used to orientate distances relative to prevailing wind direction,
        pass an xr.DataArray with zeros to get distances per absolute directions (not relative to
        prevailing wind direction)
    turbines : xr.DataSet
        as returned by load_turbines()
    bin_size_deg : float
        size of direction bins in degrees

    Returns
    -------
    xr.DataArray
        dims: turbines, direction
        direction is relative to prevail_wind_direction, i.e. 0rad = in prevailing wind direction,
        and otherwise counter-clockwise relative to 0rad

    """
    if turbines is None:
        turbines = load_turbines()

    n_bins = 360//bin_size_deg
    distances = np.ones((turbines.sizes['turbines'], n_bins)) * np.nan

    distances = xr.DataArray(distances, dims=('turbines', 'direction'))

    d = None

    iterator = zip(turbines.groupby(cluster_per_location),
                   prevail_wind_direction.groupby(cluster_per_location))

    # TODO this loop could be parallelized, but a lock is needed for writing to distances, right?
    #  how about using dask.bag.foldby? would it help to use dask.delayed to speed up the inner
    #  loop and then combine results sequential?
    for ((idx_turbine, turbines_cluster), (idx_prevail, prevail_cluster)) in iterator:
        d = calc_dist_in_direction_cluster(
            turbines_cluster,
            prevail_wind_direction=prevail_cluster,
            bin_size_deg=bin_size_deg
        )
        idcs = cluster_per_location == idx_turbine
        distances.loc[{'turbines': idcs}] = d

    if d is None:
        raise ValueError("no location found for given clusters and cluster_per_location: "
                         f"cluster_per_location={cluster_per_location}")

    # This is dangerously assuming that calc_dist_in_direction_cluster() always returns same
    # coordinates for dim=direction (which should be the case because bins and range in
    # np.histogram_bin_edges() is fixed) and that distances contains all turbines.
    distances = distances.assign_coords(direction=d.direction, turbines=turbines.turbines)

    return distances


def calc_distance_factors(turbines, distances):
    """Returns a distance factor per turbine location and direction, i.e. for each turbine and
    direction how many times its rotor diameter is the next turbine location.

    Parameters
    ----------
    turbines : xr.DataSet
        as returned by load_turbines()
    distances : xrDataArray
        as returned by calc_dist_in_direction()

    Returns
    -------
    xr.DataArray
        dims = turbines, direction
        NaN for unknown rotor diameter and if distance to next turbine is infinite

    """
    distance_factors = distances * KM_TO_METER / turbines.t_rd
    distance_factors = distance_factors.where(distance_factors < np.inf)

    return distance_factors
