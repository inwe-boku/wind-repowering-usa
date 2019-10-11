import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_toolkits import axes_grid1
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import EngFormatter

from scipy.ndimage.filters import gaussian_filter
from pandas.plotting import register_matplotlib_converters

from wind_repower_usa import turbine_models
from wind_repower_usa.calculations import calc_bounding_box_usa
from wind_repower_usa.config import DISTANCE_FACTORS, FIGSIZE
from wind_repower_usa.constants import EARTH_RADIUS_KM, METER_TO_KM
from wind_repower_usa.load_data import load_generated_energy_gwh, load_turbines
from wind_repower_usa.turbine_models import new_turbine_models, ge15_77
from wind_repower_usa.util import turbine_locations, edges_to_center

# this is actually 1 extra color, we have 7 models ATM
TURBINE_COLORS = '#000000', '#f0c220', '#fbd7a8', '#0d8085', '#c72321',


def plot_simulated_generated_energy(simulated_energy_gwh):
    register_matplotlib_converters()

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    generated_energy_gwh = load_generated_energy_gwh()

    ax.axhline(linewidth=2, color='black')

    ax.plot(generated_energy_gwh.time.values,
            generated_energy_gwh, '-', color='#c72321',
            label=generated_energy_gwh.name)

    ax.plot(simulated_energy_gwh.time.values,
            simulated_energy_gwh, '-', color='#efc220',
            label=simulated_energy_gwh.name)

    idcs = simulated_energy_gwh.time <= generated_energy_gwh.time.max()
    idcs &= simulated_energy_gwh.time >= generated_energy_gwh.time.min()
    months = simulated_energy_gwh.time.values[idcs]
    months.sort()

    generated_energy_gwh_months = generated_energy_gwh.sel(time=months)
    simulated_energy_gwh_months = simulated_energy_gwh.sel(time=months)
    ax.plot(months,
            generated_energy_gwh_months - simulated_energy_gwh_months, '-',
            label='Error (generation - simulation)', color='#0d8085')

    plt.legend()
    plt.ylabel('Wind energy generation per month [GWh]')
    plt.xlabel('Time')

    plt.grid(True)

    return plt.gca()


def plot_rel_error_energy_simulation(simulated_energy_gwh,
                                     generated_energy_gwh):
    """Plot simulation - generated energy relative to generated energy."""
    rel_error = 100. * (simulated_energy_gwh / generated_energy_gwh - 1)

    lines, = plt.plot(simulated_energy_gwh.time, rel_error, 'o-')

    sigma = 12  # pretty arbitrary value
    plt.plot(simulated_energy_gwh.time, gaussian_filter(rel_error, sigma))
    plt.legend(["Relative error",
                f"Averaged relative error (Gaussian filter, Sigma = {sigma})"])

    plt.grid(True)

    lines.figure.set_figwidth(20)
    lines.figure.set_figheight(12)

    years_locator = mdates.YearLocator()  # every year
    months_locator = mdates.MonthLocator()  # every month
    lines.figure.axes[0].xaxis.set_major_locator(years_locator)
    lines.figure.axes[0].xaxis.set_minor_locator(months_locator)

    return plt.gca()


def plot_power_curve_linearization(power_curve, linear_pc,
                                   wind_speed_probality):
    power_curve.plot()
    lines, = linear_pc.plot()
    lines.axes.plot(power_curve.wind_speed, power_curve - linear_pc)
    plt.xlabel('Wind speed [m/s]')

    plt.ylim(-300, 1650)
    plt.grid(True)
    lines.figure.set_figwidth(20)
    lines.figure.set_figheight(8)

    plt.legend(['Power curve', 'Linear power curve', 'Error of linearization'])

    ax2 = lines.axes.twinx()
    ax2.plot(wind_speed_probality.wind_speed, wind_speed_probality, 'k')

    plt.ylabel('probability at turbine locations')
    plt.legend(['Wind speed probability'])

    return plt.gca()


def plot_repower_potential(*repower_potentials, variable='power_generation', stacklabels=None):
    """This function plots either expected power generation or total number of installed
    turbines."""
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    plt.xlabel('Number of repowered turbines')

    ylabel = {
        'power_generation':           'Annual wind power generation [TWh/yr]',
        'power_gain_per_model':       'Additional annual wind power generation [TWh/yr]',
        'num_turbines': 'Total number of turbines',
    }
    factor = {
        'power_generation': 1e-3,
        'power_gain_per_model': 1e-3,
        'num_turbines': 1.,
    }

    plt.ylabel(ylabel[variable])  # FIXME make sure that this is GW!
    plt.grid(True)

    colors = TURBINE_COLORS
    turbine_names = ['mixed'] + [t.file_name for t in new_turbine_models()]
    turbine_color = dict(zip(turbine_names, colors))

    styles = ('--', '--', 'dotted', '-.', '-')
    distance_factor_style = dict(zip(DISTANCE_FACTORS + (0,), styles))
    distance_factor_style = {k: v for k, v in distance_factor_style.items()}

    distance_factors = []

    labels = []

    for repower_potential in repower_potentials:
        num_new_turbines = repower_potential.num_new_turbines

        turbine_model_name = repower_potential.attrs['turbine_model_new']
        distance_factor = repower_potential.attrs['distance_factor']
        distance_factors.append(distance_factor)
        linestyle = distance_factor_style[distance_factor]

        if turbine_model_name != 'mixed':
            turbine_model = getattr(turbine_models, turbine_model_name)
        else:
            class Turbine:
                pass  # ok, this a bit dirty...
            turbine_model = Turbine()
            turbine_model.name = 'Best turbine model per cluster'
            linestyle = '--'
        color = turbine_color[turbine_model_name]

        label = turbine_model.name if distance_factor in (2, 0) else '_nolegend_'
        if turbine_model_name != 'mixed':
            labels.append(label)

        if variable == 'power_gain_per_model':
            ax.stackplot(num_new_turbines, factor[variable] * repower_potential[variable],
                         labels=stacklabels, colors=TURBINE_COLORS[1:])
        else:
            ax.plot(num_new_turbines, factor[variable] * repower_potential[variable],
                    linestyle=linestyle, label=label, color=color)

    loc = 'upper left' if variable != 'num_turbines' else None
    legend1 = ax.legend(loc=loc)

    dist_factors = [Line2D([], [], color='black', linestyle=distance_factor_style[df],
                           label=f"Distance factor {df}") for df in distance_factors if df != 0]

    if dist_factors:
        ax.legend(handles=dist_factors, loc='upper right')
        ax.add_artist(legend1)

    return plt.gca(), labels


def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.

    Stolen from here:
    https://stackoverflow.com/a/33505522/859591
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_mean_wind_speed_and_turbines(wind_speed_mean, turbines):
    north, west, south, east = calc_bounding_box_usa(turbines)
    extent = [west, east, south, north]

    cmap = LinearSegmentedColormap.from_list('water', ['#9bdade', '#FFFFFF'])

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    wind_speed_map = ax.imshow(wind_speed_mean, origin='upper', extent=extent, cmap=cmap)
    ax.plot(turbines.xlong, turbines.ylat, '.', color='#C72321', markersize=1)

    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")

    cbar = _add_colorbar(wind_speed_map)
    cbar.ax.set_ylabel(f"Mean wind speed in [m/s]", rotation=-90, labelpad=14)

    return fig


def plot_optimized_cluster(turbines, cluster_per_location, is_optimal_location, turbine,
                           distance_factors, prevail_wind_direction, step=3):
    plot_optimal_locations = plot_wind_directions = plot_thresholds = False
    if step > 0:
        plot_optimal_locations = True
    if step > 1:
        plot_wind_directions = True
    if step > 2:
        plot_thresholds = True

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    # some arbitrary cluster with 70-100 turbines or so
    # probably: cluster=812 (but depends on clustering, therefore pinning via long/lat)
    x, y = -99.0, 45.92  # some point in cluster
    some_turbine_idx = (((turbines.xlong - x)**2 + (turbines.ylat - y)**2)**0.5).argmin()
    cluster = cluster_per_location[some_turbine_idx].values
    loc = 'upper left'

    locations = turbine_locations(turbines)
    idcs = cluster_per_location == cluster
    is_optimal_location = is_optimal_location.sum(dim='turbine_model')
    is_optimal_location = is_optimal_location.astype(np.bool)
    locations_old_ylat, locations_old_xlon = locations[idcs].T
    locations_new_ylat, locations_new_xlon = locations[idcs & is_optimal_location].T

    def radial_plot(ax, angles, radius, center, label):
        angles = np.append(angles, angles[0])
        radius = np.append(radius, radius[0])

        points_complex = center + radius * np.exp(1j * angles)
        alpha = None if plot_thresholds else 0.  # ugly hack to avoid changing figure size
        if not plot_thresholds:
            label = None
        ax.plot(points_complex.real, points_complex.imag, '-', color='gray', linewidth=0.4,
                label=label, alpha=alpha)
        return ax

    rotor_diameter = turbine.rotor_diameter_m

    has_label = False
    for idx in turbines.turbines[idcs & is_optimal_location]:
        radius = distance_factors / (EARTH_RADIUS_KM * 2 * np.pi) * 360
        radius = radius * METER_TO_KM * rotor_diameter
        center = turbines.isel(turbines=idx).xlong + turbines.isel(turbines=idx).ylat * 1j
        radial_plot(ax,
                    angles=distance_factors.direction + prevail_wind_direction.sel(turbines=idx),
                    radius=radius, center=center.values,
                    label='Minimum distance to other turbine' if not has_label else ''
                    )
        has_label = True

    ax.plot(locations_old_xlon, locations_old_ylat, 'o', markersize=4, color='#efc220',
            label='Current location of wind turbine')

    if plot_optimal_locations:
        ax.plot(locations_new_xlon, locations_new_ylat, 'x', markersize=3, color='#c72321',
                label='Optimal location for {}'.format(turbine.name))

    ax.legend(loc=loc)

    def add_arrow(label):
        # not sure why quiver key is not working
        # https://stackoverflow.com/a/22349717/859591
        from matplotlib.legend_handler import HandlerPatch
        import matplotlib.patches as mpatches

        def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True,
                                    head_width=0.5 * height)
            return p

        arrow = plt.arrow(0, 0, 1, 1, color='k')
        handles, labels = ax.get_legend_handles_labels()
        labels = labels + [label]
        handles = handles + [arrow]
        if step == 3:
            labels = labels[1:] + labels[:1]
            handles = handles[1:] + handles[:1]
        plt.legend(handles, labels,
                   handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow), },
                   loc=loc)

    if plot_wind_directions:
        ax.quiver(locations_old_xlon, locations_old_ylat,
                  np.cos(prevail_wind_direction.sel(turbines=idcs)),
                  np.sin(prevail_wind_direction.sel(turbines=idcs)),
                  width=0.0017,
                  color='k')

        add_arrow('Prevailing wind direction')

    ax.set_aspect('equal')
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")

    return fig


def plot_history_turbines():
    turbines = load_turbines()

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    # wow is this complicated to get rid of NaN-warnings by median()! isn't there an easier way?
    idcs_not_nan = {'turbines': ~np.isnan(turbines.t_cap)}
    per_year = turbines.t_cap[idcs_not_nan].groupby(turbines.p_year[idcs_not_nan])
    per_year.median(dim='turbines').plot(label='Median capacity of new turbines [kW]',
                                         ax=ax, marker='o', color='#efc220')

    ax.legend()
    plt.xlabel('Year')
    plt.ylabel('Capacity [kW]')
    plt.grid(True)

    ax2 = ax.twinx()
    idcs_not_nan = {'turbines': ~np.isnan(turbines.t_rd)}
    per_year = turbines.t_rd[idcs_not_nan].groupby(turbines.p_year[idcs_not_nan])
    per_year.median(dim='turbines').plot(label='Median rotor diameter of new turbines [m]',
                                         marker='o', color='#0d8085', ax=ax2)
    plt.ylabel("Rotor diameter [m]")
    ax2.legend(loc=1)

    return fig


def plot_min_distances(turbines, distances, title='', factors=None, quantiles=None):
    distances = distances.where(distances < np.inf)
    idcs_not_nan = ~np.isnan(turbines.t_rd)
    rotor_diameters_m = turbines.t_rd[idcs_not_nan]
    min_distances_not_nan = distances[idcs_not_nan] * 1e3
    bin_edges = np.histogram_bin_edges(rotor_diameters_m, bins=15, range=(5, 155))
    bin_idcs = np.digitize(rotor_diameters_m, bin_edges)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    idcs_cut_off = min_distances_not_nan < 500
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    x = bin_centers[bin_idcs[idcs_cut_off] - 1]
    ax = sns.stripplot(x=x, y=min_distances_not_nan[idcs_cut_off], jitter=.4, size=1, color='k')

    colors = '#c72321', '#fbd7a9', '#f0c220', '#7a6952',
    factors = factors or DISTANCE_FACTORS

    for color, factor in zip(colors, factors):
        ax.plot(factor * bin_centers, label=f'{factor:.2f}x', color=color)

    colors = '#246b71', '#6a9395', '#84bcbf', '#9bdade'
    quantiles = quantiles or (0.05, 0.1, 0.2, 0.3)
    for q, color in zip(quantiles, colors):
        ax.plot(pd.Series(min_distances_not_nan).groupby(bin_idcs).quantile(q=q).values,
                label=f"{int(q * 100)}% quantile", color=color)

    ax.set_ylim(0, 500)
    ax.set_xlim(0, 11)

    plt.ylabel('Distance to closest turbine [m]')
    plt.xlabel('Rotor diameter [m]')

    if title:
        plt.title(title)

    plt.legend()

    plt.grid()

    return fig


def plot_power_curves():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    x = np.linspace(0, 40, num=100)
    for turbine_model, color in zip((ge15_77,) + new_turbine_models(), TURBINE_COLORS):
        linestyle = 'dashed' if turbine_model == ge15_77 else '-'
        ax.plot(x, turbine_model.power_curve(x), label=turbine_model.name, color=color,
                linestyle=linestyle)
    plt.legend()
    plt.ylabel('Power generation [kW]')
    plt.xlabel('Wind speed [m/s]')

    plt.grid()

    return fig


def plot_wind_rose(data1, data2=None, percentage=True, args=None, kwargs=None, fig=None, ax=None):
    """Input in polar coordinates in mathematical orientation, but plot as wind rose.
    Mathematical orientation starts on the x-axis and is counter-clockwise, while wind rose
    starts with 0° in north direction (positive y-axis) and is clockwise oriented.

    Parameters
    ----------
    data1 : array_like, shape (N,) or (N,D)
        values to be plotted from -np.pi to np.pi in regular intervals if data2 is not given,
        otherwise interpreted as polar coordinates (i.e. 0 lies on positive x-axis, np.pi/2 on
        positive y-axis etc.)
    data2 : array_like, shape (N,) or (N,D)
        values if data1 is used for coordinates
    percentage : bool
        scale values by 100 and label with %
    args : iterable
        passed directly to ax.plot()
    kwargs : dict
        passed directly to ax.plot()
    fig : matplotlib.figure.Figure
        matplotlib figure to be used if not None
    ax : matplotlib.projections.polar.PolarAxes
        matplotlib axis to be used if not None


    Examples
    --------

    Plots a spiral starting at value 0 on the x-axis to 19 after rotating 360° in mathematical
    positive sense:

    >>> figure = plot_wind_rose(np.linspace(0, 2*np.pi, num=20), np.arange(20))

    Plots two squares with circle markers:

    >>> figure = plot_wind_rose(np.pi/2 * np.arange(5), [2,3] * np.ones((5,2)), args=('o-',))

    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    if data2 is None:
        values = data1
        directions = edges_to_center(np.linspace(-np.pi, np.pi, num=len(values) + 1))
    elif isinstance(data2, str):
        raise ValueError(f"invalid type str for data2: {data2}")
    else:
        directions = data1
        values = data2

    # close circle if first point and last are not the same, might plot points double
    directions = np.append(directions, directions[0])
    values = np.append(values, values[0:1], axis=0)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, subplot_kw=dict(polar=True))

    scale = 1.
    if percentage:
        ax.yaxis.set_major_formatter(EngFormatter(unit='%'))
        scale = 100

    # TODO there might be a 180° error in here, it is calibrated to ERA5 data and wind roses
    #  it is not entirely clear which direction it should go, would does North mean? North wind?

    ax.plot(-directions - np.pi / 2., scale * values, *args, **kwargs)
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')

    return fig, ax


def plot_locations(turbines=None, idcs=None, directions=None, colors=None):
    """Plot turbine locations and add arrows to indicate wind directions.

    FIXME does not use proper projection, probably valid only for small regions.

    Parameters
    ----------
    turbines : xr.DataSet
        as returned by load_turbines()
    idcs : array_like of type boolean
        select turbines to plot
    directions : dict of form label: array_like
        array contains directions in rad
    colors : iterable
        color of arrows for each item in directions

    """
    if turbines is None:
        turbines = load_turbines()
    if idcs is None:
        idcs = np.ones_like(turbines.xlong).astype(np.bool)
    if directions is None:
        directions = {}
    if colors is None:
        colors = [None] * len(directions)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    locations = turbine_locations(turbines.sel(turbines=idcs))

    ax.plot(locations.T[1], locations.T[0], 'o', label='Wind turbine location')

    for (label, values), color in zip(directions.items(), colors):
        ax.quiver(locations.T[1],
                  locations.T[0],
                  np.cos(values.sel(turbines=idcs)),
                  np.sin(values.sel(turbines=idcs)),
                  width=0.002,
                  label=label,
                  color=color,
                  )

    ax.set_aspect('equal')

    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.legend()

    return fig, ax
