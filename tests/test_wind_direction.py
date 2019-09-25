import numpy as np
import xarray as xr

from wind_repower_usa.constants import EARTH_RADIUS_KM
from wind_repower_usa.wind_direction import calc_wind_rose
from wind_repower_usa.wind_direction import calc_directions
from wind_repower_usa.wind_direction import calc_dist_in_direction


def test_calc_wind_rose():
    num_turbines = 100
    turbines = xr.Dataset({
        'xlong': ('turbines', np.arange(num_turbines, dtype=np.float64)),
        'ylat': ('turbines', np.arange(num_turbines, dtype=np.float64)),
        't_cap': ('turbines', np.ones(num_turbines)),
    },
        coords={'turbines': np.arange(13, num_turbines + 13)}
    )

    num_time_stamps = 17
    wind_speed = xr.DataArray(np.ones((num_time_stamps, num_turbines)),
                              dims=('time', 'turbines'),
                              coords={'time': np.arange(num_time_stamps),
                                      'longitude': turbines.xlong,
                                      'latitude': turbines.ylat})

    wind_velocity_array = np.ones((num_time_stamps, num_turbines + 20, num_turbines + 25),
                                  dtype=np.float32)  # ERA5 data comes in 32bit format!
    wind_velocity = xr.Dataset({
        'u100': (('time', 'latitude', 'longitude'), wind_velocity_array),
        'v100': (('time', 'latitude', 'longitude'), wind_velocity_array),
        'u10': (('time', 'latitude', 'longitude'), wind_velocity_array),
        'v10': (('time', 'latitude', 'longitude'), wind_velocity_array),
    },
        coords={
            'latitude': np.arange(-10, num_turbines + 10),
            'longitude': np.arange(-10, num_turbines + 15),
            'time': np.arange(17),
        }
    )

    directivity_width = 1

    wind_rose, prevail_wind_direction, directivity = calc_wind_rose(
        turbines, wind_speed, wind_velocity, power_curve=lambda x: x,
        bins=80, directivity_width=directivity_width)

    # FIXME weirdly setting bins=100 or to 90 leads to a different error than the expected one...
    max_error = (2 * np.pi / 80 / 2) / (np.pi / 4)

    np.testing.assert_allclose(prevail_wind_direction, np.pi / 4, rtol=max_error)

    assert np.all(wind_rose.turbines == turbines.turbines)

    # relative rations should integrate to 1 because density=True in np.histogram()
    np.testing.assert_allclose(wind_rose.integrate('direction'), 1)


def test_calc_directions():
    num_turbines = 10
    turbines = xr.Dataset({
        'xlong': ('turbines', 1e-4 * np.arange(num_turbines, dtype=np.float64)),
        'ylat': ('turbines', 1e-4 * np.arange(num_turbines, dtype=np.float64)),
    },
        coords={'turbines': np.arange(13, num_turbines + 13)}
    )

    prevail_wind_direction = xr.DataArray(np.zeros(num_turbines), dims='turbines',
                                          coords={'turbines': turbines.turbines})

    directions = calc_directions(turbines, prevail_wind_direction)

    assert directions.dims == ('targets', 'turbines')
    np.testing.assert_array_equal(directions.values[np.diag_indices_from(directions.values)],
                                  np.nan * np.ones(num_turbines))

    assert directions.isel(turbines=0, targets=1) == np.pi/4
    assert directions.isel(turbines=3, targets=1) == -3/4 * np.pi


def test_calc_dist_in_direction_turbines_in_a_row():
    num_turbines = 10
    turbines = xr.Dataset({
        'xlong': ('turbines', 1e-4 * np.arange(num_turbines, dtype=np.float64)),
        'ylat': ('turbines', 1e-4 * np.arange(num_turbines, dtype=np.float64)),
    },
        coords={'turbines': np.arange(13, num_turbines + 13)}
    )

    cluster_per_location = xr.DataArray(np.zeros(num_turbines), dims='turbines',
                                        name='cluster_per_location')
    prevail_wind_direction = xr.DataArray(np.zeros(num_turbines), dims='turbines',
                                          coords={'turbines': turbines.turbines})

    distances = calc_dist_in_direction(cluster_per_location,
                                       prevail_wind_direction,
                                       turbines=turbines,
                                       bin_size_deg=30)

    assert len(distances.direction) == 360/30

    dist = 2 * np.pi * 1e-4 / 360 * EARTH_RADIUS_KM * 2 ** .5
    np.testing.assert_allclose(distances.isel(direction=[1, 7], turbines=slice(1, -1)),
                               dist)
    np.testing.assert_allclose(distances.isel(direction=[1, 7], turbines=[0, -1]),
                               [[np.inf, dist],
                                [dist, np.inf]])
    np.testing.assert_allclose(distances.isel(direction=[1, 7]).direction * 180 / np.pi,
                               [-135, 45])


def test_calc_dist_in_direction_turbines_in_a_grid():
    num_turbines = 12
    x = 1e-4 * np.arange(3, dtype=np.float64)
    y = 1e-4 * np.array([0, 1, 4, 100])
    xx, yy = np.meshgrid(x, y)
    turbines = xr.Dataset({
        'xlong': ('turbines', xx.flatten()),
        'ylat': ('turbines', yy.flatten()),
    },
        coords={'turbines': np.arange(13, num_turbines + 13)}
    )

    cluster_per_location = xr.DataArray(np.zeros(num_turbines), dims='turbines',
                                        name='cluster_per_location')
    cluster_per_location[-3:] = 1
    prevail_wind_direction = xr.DataArray(np.zeros(num_turbines), dims='turbines',
                                          coords={'turbines': turbines.turbines})

    distances = calc_dist_in_direction(cluster_per_location,
                                       prevail_wind_direction,
                                       turbines=turbines,
                                       bin_size_deg=45)

    assert len(distances.direction) == 360//45

    dist = 2 * np.pi * 1e-4 / 360 * EARTH_RADIUS_KM
    # different cluster, straight line - only direct neighbours have non-inf distances
    assert np.all(distances.isel(direction=slice(1, 4), turbines=slice(-3, None)) == np.inf)
    assert np.all(distances.isel(direction=slice(5, None), turbines=slice(-3, None)) == np.inf)

    central_turbine = turbines.isel(turbines=(turbines.xlong == 1e-4) & (turbines.ylat == 1e-4))
    np.testing.assert_allclose(distances.sel(turbines=central_turbine.turbines),
                               [[dist, dist * 2**.5, dist, dist * 2**.5,
                                dist, dist * 10**.5, 3 * dist, np.inf]])
