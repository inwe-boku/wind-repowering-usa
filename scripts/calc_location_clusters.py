import logging

from wind_repower_usa.config import DISTANCE_FACTORS, INTERIM_DIR, COMPUTE_CONSTANT_DISTANCE_FACTORS
from wind_repower_usa.constants import METER_TO_KM
from wind_repower_usa.geographic_coordinates import calc_location_clusters
from wind_repower_usa.load_data import load_turbines, load_distance_factors
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.turbine_models import new_turbine_models


setup_logging()

logging.info("Start clustering turbine locations...")
distance_factors = load_distance_factors()
turbines = load_turbines()

# let's assume that the largest model is amongst new_turbine_models()...
max_rotor_diameter_m = max(tm.rotor_diameter_m for tm in new_turbine_models())

df = (0,)
if COMPUTE_CONSTANT_DISTANCE_FACTORS:
    df += DISTANCE_FACTORS

for distance_factor in df:
    logging.info(f"Clustering for distance factor: {distance_factor}...")
    if distance_factor == 0:
        df = distance_factors.max()
        df_filename = ''
    else:
        df = distance_factor
        df_filename = f'_{distance_factor}'

    min_distance_km = df * max_rotor_diameter_m * METER_TO_KM
    cluster_per_location, _, _ = calc_location_clusters(turbines, min_distance_km)

    cluster_per_location.attrs['distance_factor'] = distance_factor  # 0 for direction dependent
    cluster_per_location.attrs['min_distance_km'] = float(min_distance_km)

    cluster_per_location.to_netcdf(INTERIM_DIR / 'optimal_locations' /
                                   f'cluster_per_location{df_filename}.nc')

logging.info("Done!")
