import json

import pandas as pd
import xarray as xr

from wind_repower_usa.config import INTERIM_DIR, EXTERNAL_DIR
from wind_repower_usa.turbine_models import ge15_77

NUM_TURBINES = 58000


def load_turbines():
    """Load list of all turbines from CSV file. Includes location, capacity,
    etc. Missing values are replaced with NaN values.

    The file uswtdb_v1_2_20181001.xml contains more information about the fields.

    Returns
    -------
    xr.DataSet

    """
    turbines_dataframe = pd.read_csv(EXTERNAL_DIR / 'wind_turbines_usa' /
                                     'uswtdb_v1_3_20190107.csv')

    # TODO is this really how it is supposed to be done?
    turbines_dataframe.index = turbines_dataframe.index.rename('turbines')
    turbines = xr.Dataset.from_dataframe(turbines_dataframe)

    # Lets not use the turbine on Guam (avoids a huge bounding box for the USA)
    turbines = turbines.sel(turbines=turbines.xlong < 0)

    # turbine coordinates should be sequential also after removing Guam...
    turbines['turbines'] = range(turbines.turbines.size)

    return turbines


def load_generated_energy_gwh():
    with open(EXTERNAL_DIR / 'energy_generation' / 'ELEC.GEN.WND-US-99.M.json', 'r') as f:
        generated_energy_json = json.load(f)

    date, value = zip(*generated_energy_json['series'][0]['data'])

    # unit = thousand megawatthours
    generated_energy_gwh = pd.Series(value, index=pd.to_datetime(date, format='%Y%m'))

    return xr.DataArray(generated_energy_gwh, dims='time',
                        name="Generated energy per month [GWh]")


def load_wind_velocity(year, month):
    """month/year can be list or int"""
    try:
        iter(year)
    except TypeError:
        year = [year]

    try:
        iter(month)
    except TypeError:
        month = [month]

    fnames = [EXTERNAL_DIR / 'wind_velocity_usa_era5' /
              'wind_velocity_usa_{y}-{m:02d}.nc'.format(m=m, y=y)
              for m in month for y in year]

    # TODO how to calculate this better? Could be performance or RAM issue!
    chunk_size_total = 1e6
    time_chunk_size = int(chunk_size_total / NUM_TURBINES)

    wind_velocity_datasets = [
        xr.open_dataset(fname,
                        chunks={'time': time_chunk_size})
        for fname in fnames]

    wind_velocity = xr.concat(wind_velocity_datasets, dim='time')

    return wind_velocity


def load_wind_speed(years, months):
    """Load wind speed from processed data files.

    Parameters
    ----------
    years : int or list of ints
    months : int or list of ints

    Returns
    -------
    xr.DataArray

    """
    try:
        iter(years)
    except TypeError:
        years = [years]

    try:
        iter(months)
    except TypeError:
        months = [months]

    fnames = [INTERIM_DIR / 'wind_speed_usa_era5' /
              'wind_speed_usa_era5-{}-{:02d}.nc'.format(year, month)
              for year in years for month in months]

    # TODO how to calculate this better? Could be performance or RAM issue!
    chunk_size_total = 1e6
    time_chunk_size = int(chunk_size_total / NUM_TURBINES)

    wind_speed = xr.open_mfdataset(fnames,
                                   chunks={'time': time_chunk_size})

    if len(wind_speed.data_vars) != 1:
        raise ValueError("This is not a DataArray")

    return wind_speed.__xarray_dataarray_variable__


def load_optimal_locations(turbine_model, distance_factor):
    df_filename = '' if distance_factor is None else f'_{distance_factor}'
    is_optimal_location = xr.open_dataarray(
        INTERIM_DIR / 'optimal_locations' /
        f'is_optimal_location_{turbine_model.file_name}{df_filename }.nc')
    return is_optimal_location


def load_simulated_energy_per_location(turbine_model, capacity_scaling=False):
    scaling_str = '' if not capacity_scaling else '_capacity_scaled'
    simulated_energy_per_location = xr.open_dataarray(
        INTERIM_DIR / 'simulated_energy_per_location' /
        f'simulated_energy_{turbine_model.file_name}{scaling_str}_gwh.nc')
    return simulated_energy_per_location


def load_repower_potential(turbine_model_new, distance_factor):
    turbine_model_old = ge15_77
    if turbine_model_new == 'mixed':
        turbine_model_new_fname = 'mixed'
    else:
        turbine_model_new_fname = turbine_model_new.file_name
    df_filename = '' if distance_factor is None else f'_{distance_factor}'
    fname = (INTERIM_DIR / 'repower_potential' /
             f'repower_potential_{turbine_model_old.file_name}_'
             f'{turbine_model_new_fname}{df_filename }.nc')

    return xr.open_dataset(fname)


def load_distances(relative=True):
    relabs = 'relative' if relative else 'absolute'
    return xr.open_dataarray(INTERIM_DIR / 'distances_in_direction' / f'distances-{relabs}.nc')


def load_prevail_wind_direction():
    prevail_wind_direction = xr.open_dataarray(
        INTERIM_DIR / 'wind-direction' / 'prevail_wind_direction.nc')
    return prevail_wind_direction


def load_distance_factors():
    return xr.open_dataarray(INTERIM_DIR / 'distances_in_direction' / 'distance_factors.nc')


def load_cluster_per_location(distance_factor):
    df_filename = '' if distance_factor is None else f'_{distance_factor}'
    return xr.open_dataarray(INTERIM_DIR / 'optimal_locations' /
                             f'cluster_per_location{df_filename}.nc')
