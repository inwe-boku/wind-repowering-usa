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
