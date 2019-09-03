import numpy as np
import xarray as xr
from wind_repower_usa.wind_direction import calc_wind_rose


def test_calc_wind_rose():
    num_turbines = 100
    turbines = xr.Dataset({
        'xlong': ('turbines', np.arange(num_turbines, dtype=np.float64)),
        'ylat': ('turbines', np.arange(num_turbines, dtype=np.float64)),
        't_cap': ('turbines', np.ones(num_turbines, dtype=np.float64)),
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
                                  dtype=np.float32)
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
