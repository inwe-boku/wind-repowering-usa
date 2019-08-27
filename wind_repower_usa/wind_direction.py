import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from wind_repower_usa.calculations import calc_simulated_energy


# TODO add turbine model?
from wind_repower_usa.geographic_coordinates import geolocation_distances
from wind_repower_usa.util import turbine_locations


def calc_wind_rose(turbines, wind_speed, wind_velocity, bins=70):
    """Calculate prevailing wind direction for each turbine location in ``turbines``.

    Parameters
    ----------
    turbines : xr.Dataset
    wind_speed :
    wind_velocity :
        as downloaded from ERA5
    bins : int
        bins for histogram of distribution of energy (~wind speed) over direction

    Returns
    -------
    hists : ???
    prevail_wind_direction : float
        direction in rad, between -np.pi and np.pi, zero is east, np.pi/2 is north
    directivity : float


    """
    # TODO speed up potential of this function:
    #  - choose some wind speed samples only
    #  - calculate only once for entire park

    # interpolation is already done, but only stored as wind speed, u/v components not separately
    wind_velocity_at_turbines = wind_velocity.interp(
        longitude=xr.DataArray(turbines.xlong.values, dims='turbines'),
        latitude=xr.DataArray(turbines.ylat.values, dims='turbines'),
        method='linear')

    # FIXME do we need a special azimuth formula here? Does this depend on projection?
    # TODO u is east, v north, right?
    # http://www.wikience.org/documentation/wind-speed-and-direction-tutorial/
    directions = np.arctan2(wind_velocity_at_turbines.v100,
                            wind_velocity_at_turbines.u100).compute()

    energy = calc_simulated_energy(wind_speed,
                                   turbines,
                                   sum_along='',
                                   only_built_turbines=False)

    hists = []

    for turbine_idx in range(directions.values.shape[1]):
        hist = np.histogram(directions.values.T[turbine_idx, :],
                            weights=energy.values.T[turbine_idx, :], bins=bins, density=True)
        hists.append(hist)

    # FIXME warning: from here on last hist is used

    # +/- 20° angle
    boxcar_width_angle = np.radians(25)
    boxcar_width = int(np.round(boxcar_width_angle / (2 * np.pi / len(hist[1]))))

    convoluted = uniform_filter(hist[0], boxcar_width)

    prevail_wind_direction = hist[1][np.argmax(convoluted)]

    # FIXME this needs somehow to be devided by box_carwidth or so?
    directivity = np.max(convoluted) * boxcar_width_angle

    return hists, prevail_wind_direction, directivity


def calc_dist_in_direction(turbines, prevail_wind_direction, bin_size_deg=15):
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

    bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.
    direction_bins = xr.DataArray(np.arange(num_bins), dims='direction',
                                  coords={'direction': bin_centers * 180 / np.pi})

    return xr.where(bin_idcs == direction_bins, distances, np.inf).min(dim='targets')
