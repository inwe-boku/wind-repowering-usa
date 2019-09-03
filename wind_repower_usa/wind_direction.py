import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from wind_repower_usa.calculations import calc_simulated_energy
from wind_repower_usa.geographic_coordinates import geolocation_distances
from wind_repower_usa.load_data import load_turbines
from wind_repower_usa.util import turbine_locations, edges_to_center


def calc_wind_rose(turbines, wind_speed, wind_velocity, power_curve=None, bins=70,
                   directivity_width=15):
    """Calculate prevailing wind direction for each turbine location in ``turbines``.

    Parameters
    ----------
    turbines : xr.Dataset
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

    return wind_rose, np.array(prevail_wind_direction), np.array(directivity)


def calc_dist_in_direction_cluster(turbines, prevail_wind_direction, bin_size_deg=15):
    """Directions between 0° and 360° will be grouped into bins of size ``bin_size_deg``,
    then for each turbine location the distance to the next turbine is calculated for each
    direction bin.

    Parameters
    ----------
    turbines : xr.Dataset
    prevail_wind_direction : float
        in rad, 0rad = east, np.pi = north
    bin_size_deg : float

    Returns
    -------
    xr.DataArray
        dims: turbines, direction
        direction is relative to prevail_wind_direction, i.e. 0° = in prevailing wind direction,
        and otherwise counter-clockwise relative to 0°

    """
    x1, x2 = np.meshgrid(turbines.xlong, turbines.xlong)
    y1, y2 = np.meshgrid(turbines.ylat, turbines.ylat)

    # pairwise directions from each turbine to each other one
    directions = np.arctan2(y1 - y2, x1 - x2) - prevail_wind_direction
    directions = (directions + np.pi) % (2 * np.pi) - np.pi

    # directions is actually not used here (except for dtype)
    bin_edges = np.histogram_bin_edges(directions,
                                       bins=360//bin_size_deg,
                                       range=(-np.pi, np.pi))

    num_bins = len(bin_edges) - 1  # Attention, fencepost problem!

    # np.digitize does not return the n-th bin, but the n+1-th bin!
    bin_idcs = np.digitize(directions, bin_edges) - 1
    bin_idcs = xr.DataArray(bin_idcs, dims=('turbines', 'targets'))

    locations = turbine_locations(turbines)

    distances = geolocation_distances(locations)

    # set distance to itself to INF to avoid zero distance minimums later
    distances[np.diag_indices_from(distances)] = np.inf
    distances = xr.DataArray(distances, dims=('turbines', 'targets'))

    bin_centers = edges_to_center(bin_edges)
    direction_bins = xr.DataArray(np.arange(num_bins), dims='direction',
                                  coords={'direction': bin_centers})

    return xr.where(bin_idcs == direction_bins, distances, np.inf).min(dim='targets')


def calc_dist_in_direction(clusters, cluster_per_location, turbines=None):
    if turbines is None:
        turbines = load_turbines()

    n_bins = 24
    distances = np.ones((turbines.sizes['turbines'], n_bins)) * np.nan

    # TODO missing coords for direction or so? also for turbines?
    distances = xr.DataArray(distances, dims=('turbines', 'direction'))

    for cluster in clusters[1:]:
        idcs = cluster_per_location == cluster

        # FIXME do something about NaNs in turbines.t_rd, exclude in idcs?
        # FIXME should pass through all parameters for calc_dist_in_direction_cluster()!
        d = calc_dist_in_direction_cluster(turbines.sel(turbines=idcs),
                                           prevail_wind_direction=0.,
                                           bin_size_deg=15)
        distances.loc[{'turbines': idcs}] = d

    return distances