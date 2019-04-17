import time
import logging

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from wind_repower_usa.config import MONTHS
from wind_repower_usa.load_data import load_turbines, load_wind_velocity
from wind_repower_usa.load_data import load_wind_speed
from wind_repower_usa.turbine_models import ge15_77


def calc_wind_speed_at_turbines(wind_velocity, turbines):
    # interpolate at turbine locations
    wind_velocity_at_turbines = wind_velocity.interp(
        longitude=xr.DataArray(turbines.xlong.values, dims='turbines'),
        latitude=xr.DataArray(turbines.ylat.values, dims='turbines'),
        method='linear')

    # velocity --> speed
    wind_speed = (wind_velocity_at_turbines.u100**2
                  + wind_velocity_at_turbines.v100**2)**0.5

    return wind_speed


def calc_simulated_energy(wind_speed, turbines, power_curve=None, sum_along='turbines',
                          capacity_scaling=True, only_built_turbines=True):
    """Estimate generated energy using wind data and turbine data.

    Parameters
    ----------
    wind_speed : xr.DataArray
        see calc_wind_speed_at_turbines()
    turbines : xr.DataSet
        see load_turbines()
    power_curve : callable
        a function mapping wind speed to power
    sum_along : str
        sum along turbines or time
    capacity_scaling

    Returns
    -------
    simulated_energy_gwh : xr.DataArray
        Simulated energy per month [GWh], dims = (time, turbines)

    """
    if power_curve is None:
        power_curve = ge15_77.power_curve

    # TODO this is a bit scary, when does parallelized not work? Which dtype?
    simulated_energy = xr.apply_ufunc(power_curve, wind_speed,
                                      dask='parallelized',
                                      output_dtypes=[np.float64])

    simulated_energy = simulated_energy.assign_coords(turbines=turbines.turbines)

    if only_built_turbines:
        # TODO all turbines where year = NaN will be removed that way... :-/
        # this is the beginning of the year the turbine has been commissioned
        building_dates = turbines.p_year.astype(int).astype(str).astype(np.datetime64)

        nanosecs_of_year = (simulated_energy.time - building_dates).astype(np.float)
        proportion_of_year = nanosecs_of_year / (365.25 * 24 * 60 * 60 * 1e9)
        building_this_year = simulated_energy.time.dt.year == turbines.p_year

        simulated_energy = simulated_energy.where(~building_this_year,
                                                  simulated_energy * proportion_of_year)

        already_built = simulated_energy.time.dt.year >= turbines.p_year
        simulated_energy = simulated_energy.where(already_built, 0)

        # Uargh... there is a weired memory leak somewhere, this seems to help a bit at least... :-/
        del nanosecs_of_year
        del proportion_of_year
        del building_this_year
        del building_dates
        del already_built

    if capacity_scaling:
        simulated_energy *= (turbines.t_cap / 1500.).fillna(1.)

    # inspired by:
    # http://xarray.pydata.org/en/stable/examples/weather-data.html#monthly-averaging

    simulated_energy = simulated_energy.sortby('time')

    simulated_energy = simulated_energy.sum(dim=sum_along) * 1e-6
    if sum_along == 'turbines':
        simulated_energy = simulated_energy.resample(time='1MS').sum()

    # Does not work for multiple years:
    # simulated_energy = simulated_energy.sum(dim='turbines').groupby('time.month').sum() * 1e-6

    with ProgressBar():
        simulated_energy_gwh = simulated_energy.compute()

    if sum_along == 'turbines':
        simulated_energy_gwh.name = "Simulated energy per month [GWh]"
    elif sum_along == 'time':
        simulated_energy_gwh.name = "Simulated energy"  # TODO unit depends on time range?

    return simulated_energy_gwh


def calc_simulated_energy_years(years, turbines=None, power_curve=None, capacity_scaling=True,
                                only_built_turbines=True):
    """Load wind speed data from processed files and calculate energy in a loop to avoid large
    data in memory for the parts where dask chunks are not correctly working yet.

    Parameters
    ----------
    years : iterable
        eg. range(2004, 2017)
    turbines : xr.DataSet
        as returned by load_turbines()
    power_curve : callable
        see calc_simulated_energy()

    Returns
    -------
    xr.DataArray

    """
    eta = False
    simulated_energy_gwh = []
    if turbines is None:
        turbines = load_turbines()

    with ProgressBar():
        for year in years:
            for month in range(1, 13):
                t0 = time.time()
                logging.info("Calculating {}-{:02d}...".format(year, month))
                wind_speed = load_wind_speed(year, month)
                simulated_energy_gwh.append(
                    calc_simulated_energy(
                        wind_speed=wind_speed,
                        turbines=turbines,
                        power_curve=power_curve,
                        capacity_scaling=capacity_scaling,
                        only_built_turbines=only_built_turbines
                    )
                )

                if not eta:
                    logging.info("ETA: %s seconds", (time.time() - t0) * 12 * len(years))
                    eta = True

    simulated_energy_gwh = xr.concat(simulated_energy_gwh, dim='time')
    return simulated_energy_gwh


def calc_bounding_box_usa(turbines, extension=1.):
    # Bounding box can be also manually selected:
    #   https://boundingbox.klokantech.com/

    # assert -180 <= long <= 180, -90 <= lat <= 90
    # FIXME need +180 modulo 360!
    north = turbines.ylat.values.max() + extension
    west = turbines.xlong.values.min() - extension
    south = turbines.ylat.values.min() - extension
    east = turbines.xlong.values.max() + extension

    return north, west, south, east


def calc_wind_speed_probablity(num_samples=200, year=2016, bins=150):
    """Take random wind speed samples in given years at all wind speed
    locations and return the probability density function.
    Returns
    -------
    wind_speed_probability : xr.DataArray
        wind_speed in m/s vs probability

    """
    # TODO code duplication with data_exploration notebook
    wind_speed = xr.concat([load_wind_speed(year, month)
                            for month in MONTHS], dim='time')

    # FIXME it would be choose in 2d over turbines/time
    idcs_time = np.random.choice(wind_speed.time.shape[0],
                                 size=num_samples)
    idcs_turbines = np.random.choice(wind_speed.turbines.shape[0],
                                     size=num_samples)

    wind_speed_values = wind_speed.isel(time=idcs_time,
                                        turbines=idcs_turbines).values

    probabilities_hist, wind_speed_hist = np.histogram(
        wind_speed_values.flatten(), density=True, bins=bins)

    # there are more edges of bins than values --> picket fence error
    wind_speed_hist = wind_speed_hist[1:] - np.diff(wind_speed_hist)/2.

    wind_speed_probality = xr.DataArray(probabilities_hist, dims='wind_speed',
                                        coords={'wind_speed': wind_speed_hist})
    return wind_speed_probality


def calc_mean_wind_speed(years, sample_size=200):
    wind_velocity = load_wind_velocity(years, MONTHS)
    idcs_time = np.random.choice(wind_velocity.time.shape[0],
                                 size=sample_size, replace=False)

    wind_speed_mean = ((wind_velocity.v10**2 +
                        wind_velocity.u10**2)**0.5).isel(time=idcs_time).mean(dim='time')

    return wind_speed_mean
