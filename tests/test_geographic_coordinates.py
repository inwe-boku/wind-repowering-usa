import numpy as np

from wind_repower_usa.geographic_coordinates import geolocation_distances, calc_min_distances


LOCATIONS = np.array([
    [48.2323, 110.223],
    [49.2323, 112.223],
    [48.2423, 110.233],
])


def test_geolocation_distances():
    # 0 <--> 1: 184.05km
    # 1 <--> 2: 182.78km
    # 0 <--> 2: 1.34km
    distances = geolocation_distances(np.array(LOCATIONS))
    np.testing.assert_allclose(distances,
                               [[0., 184.05, 1.34],
                                [184.05, 0., 182.78],
                                [1.34, 182.78, 0.]],
                               atol=10e-3)


def test_calc_min_distances():
    cluster_per_location = [0, 0, 0]
    min_distances = calc_min_distances(LOCATIONS, cluster_per_location)
    np.testing.assert_allclose(min_distances, np.array([1.34, 182.78, 1.34]), atol=10e-3)


def test_calc_min_distances_multiple_clusters():
    cluster_per_location = [0, 2, 0]
    min_distances = calc_min_distances(LOCATIONS, cluster_per_location)
    np.testing.assert_allclose(min_distances, np.array([1.34, np.inf, 1.34]), atol=10e-3)
