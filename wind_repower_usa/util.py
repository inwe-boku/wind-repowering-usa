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


def quantile(a, q, dim=None, **kwargs):
    """Compute the qth quantile of the data along the specified dimension. This is just a wrapper
    around `xr.DataArray.quantile()` which allows usage of `np.inf`. Not quite sure if this is
    done in a numerical stable way.
    """
    if 'interpolation' in kwargs:
        raise ValueError

    lower = a.quantile(q, dim=dim, interpolation='lower', **kwargs)
    higher = a.quantile(q, dim=dim, interpolation='higher', **kwargs)

    size = a.sizes[dim] if dim is not None else a.size
    fraction = np.fmod(q * (size - 1), 1)

    if fraction == 0.:
        return lower
    elif fraction == 0.:
        return higher
    else:
        return (1. - fraction) * lower + fraction * higher


def choose_samples(*objs, num_samples, dim):
    """Pick random samples from xarray objects.

    Parameters
    ----------
    objs : xr.DataArray or xr.Dataset
    num_samples
    dim

    Returns
    -------

    """
    assert np.all(objs[0].sizes[dim] == np.array([obj.sizes[dim] for obj in objs])), \
        f"objs must have same length in dimension {dim}, " \
        f"sizes are: {list(obj.sizes[dim] for obj in objs)}"

    np.random.seed(42)
    idcs = np.random.choice(objs[0].sizes[dim], size=num_samples)
    idcs.sort()
    return (obj.isel({dim: idcs}) for obj in objs)


def is_monotone(a, increasing=True, strict=True):
    assert isinstance(a, np.ndarray), "pass an numpy object, xrarray can make troubles with coords"
    if increasing:
        if strict:
            return np.all(a[:-1] < a[1:])
        else:
            return np.all(a[:-1] <= a[1:])
    else:
        if strict:
            return np.all(a[:-1] > a[1:])
        else:
            return np.all(a[:-1] >= a[1:])
