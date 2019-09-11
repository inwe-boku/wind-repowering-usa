import numpy as np


def turbine_locations(turbines):
    """Extract a numpy array of turbine locations from the turbine DataArray.

    Parameters
    ----------
    turbines : xr.DataArray
        as returned by load_turbines()

    Returns
    -------
    np.ndarray with shape (N, 2)

    """
    turbines_np = np.column_stack([
        turbines.ylat.values,
        turbines.xlong.values,
    ])

    return turbines_np


def edges_to_center(edges):
    return edges[:-1] + (edges[1] - edges[0])/2.


def iterate_clusters(clusters, cluster_per_location):
    """Iterate through each cluster, yielding boolean indices of included turbines and the
    cluster index. Non clustered turbines with idx == -1 are excluded.

    Parameters
    ----------
    clusters
    cluster_per_location

    """
    for cluster in clusters:
        if cluster == -1:
            # -1 is the category for all single-turbine clusters
            continue

        idcs = cluster_per_location == cluster

        if idcs.sum() == 0:
            raise RuntimeError(f"cluster {cluster} does not have any locations")

        yield idcs, cluster
