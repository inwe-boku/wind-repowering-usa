import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_toolkits import axes_grid1
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from scipy.ndimage.filters import gaussian_filter
from pandas.plotting import register_matplotlib_converters

from wind_repower_usa import turbine_models
from wind_repower_usa.calculations import calc_bounding_box_usa
from wind_repower_usa.config import DISTANCE_FACTORS, FIGSIZE
from wind_repower_usa.load_data import load_generated_energy_gwh, load_turbines
from wind_repower_usa.turbine_models import new_turbine_models, ge15_77


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
    plt.xlabel('time')

    plt.grid(True)

    return plt.gca()


def plot_rel_error_energy_simulation(simulated_energy_gwh,
                                     generated_energy_gwh):
    """Plot simulation - generated energy relative to generated energy."""
    rel_error = 100. * (simulated_energy_gwh/generated_energy_gwh - 1)

    lines, = plt.plot(simulated_energy_gwh.time, rel_error, 'o-')

    sigma = 12  # pretty arbitrary value
    plt.plot(simulated_energy_gwh.time, gaussian_filter(rel_error, sigma))
    plt.legend(["Relative error",
                f"Averaged relative error (Gaussian filter, Sigma = {sigma})"])

    plt.grid(True)

    lines.figure.set_figwidth(20)
    lines.figure.set_figheight(12)

    years_locator = mdates.YearLocator()    # every year
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


def plot_repower_potential(*repower_potentials, variable='power_generation'):
    """This function plots either expected power generation for variabel='power' or """
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    plt.xlabel('Number of repowered turbines')

    labels = {
        'power_generation': 'Average wind power generation [GW]',
        'num_turbines': 'Total number of turbines',
    }

    plt.ylabel(labels[variable])  # FIXME make sure that this is GW!
    plt.grid(True)

    colors = '#c72321', '#0d8085', '#efc220'
    turbine_names = (t.file_name for t in new_turbine_models())
    turbine_color = dict(zip(turbine_names, colors))

    styles = ('-', '--', 'dotted', '-.')
    distance_factor_style = dict(zip(DISTANCE_FACTORS, styles))

    for repower_potential in repower_potentials:
        num_new_turbines = repower_potential.num_new_turbines
        power_generation = repower_potential.power_generation

        turbine_model_name = repower_potential.attrs['turbine_model_new']
        distance_factor = repower_potential.attrs['distance_factor']
        turbine_model = getattr(turbine_models, turbine_model_name)
        color = turbine_color[turbine_model_name]

        label = turbine_model.name if distance_factor == 2 else '_nolegend_'

        y = {
            # FIXME this division should be elsewhere
            'power_generation': power_generation/365/24,
            'num_turbines': repower_potential.num_turbines
        }
        ax.plot(num_new_turbines, y[variable], linestyle=distance_factor_style[distance_factor],
                label=label, color=color)

    legend1 = ax.legend(loc='upper right')

    dist_factors = [Line2D([], [], color='black', linestyle=distance_factor_style[df],
                           label=f"Distance factor {df}") for df in DISTANCE_FACTORS]
    ax.legend(handles=dist_factors, loc='upper left')
    ax.add_artist(legend1)

    return plt.gca()


def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.

    Stolen from here:
    https://stackoverflow.com/a/33505522/859591
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_mean_wind_speed_and_turbines(wind_speed_mean, turbines):
    north, west, south, east = calc_bounding_box_usa(turbines)
    extent = [west, east, south, north]

    cmap = LinearSegmentedColormap.from_list('water', ['#273738', '#246b71', '#6a9395',
                                                       '#84bcbf', '#9bdade'])

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    wind_speed_map = ax.imshow(wind_speed_mean, origin='upper', extent=extent, cmap=cmap)
    ax.plot(turbines.xlong, turbines.ylat, '.', color='#C72321', markersize=1)

    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")

    cbar = _add_colorbar(wind_speed_map)
    cbar.ax.set_ylabel(f"Mean wind speed in [m/s]", rotation=-90, labelpad=14)

    return fig


def plot_optimized_cluster(locations, optimal_locations, turbine):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    cluster = 2924
    idcs = optimal_locations.cluster_per_location == cluster
    is_optimal_location = optimal_locations.is_optimal_location.astype(np.bool)
    locations_old_ylat, locations_old_xlon = locations[idcs].T
    locations_new_ylat, locations_new_xlon = locations[idcs & is_optimal_location].T

    ax.plot(locations_old_xlon, locations_old_ylat, 'o', markersize=2, color='#efc220',
            label='Current location of wind turbine')
    ax.plot(locations_new_xlon, locations_new_ylat, 'o', markersize=2, color='#c72321',
            label='Optimal location for {}'.format(turbine.name))
    ax.set_aspect('equal')
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    ax.legend()

    return fig


def plot_history_turbines():
    turbines = load_turbines()
    per_year = turbines.groupby('p_year')

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    per_year.median().t_cap.plot(label='Median capacity of new turbines [kW]',
                                 ax=ax, marker='o', color='#efc220')

    ax.legend()
    plt.xlabel('Year')
    plt.ylabel('Capacity [kW]')
    plt.grid(True)

    ax2 = ax.twinx()
    per_year.median().t_rd.plot(label='Median rotor diameter of new turbines [m]', marker='o',
                                color='#0d8085', ax=ax2)
    plt.ylabel("Rotor diameter [m]")
    ax2.legend(loc=1)

    return fig


def plot_min_distances(turbines, min_distances):
    idcs_not_nan = ~np.isnan(turbines.t_rd)
    rotor_diameters_m = turbines.t_rd[idcs_not_nan]
    min_distances_not_nan = min_distances[idcs_not_nan] * 1e3
    bin_edges = np.histogram_bin_edges(rotor_diameters_m, bins=15, range=(5, 155))
    bin_idcs = np.digitize(rotor_diameters_m, bin_edges)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    idcs_cut_off = min_distances_not_nan < 500
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
    x = bin_centers[bin_idcs[idcs_cut_off] - 1]
    ax = sns.stripplot(x=x, y=min_distances_not_nan[idcs_cut_off], jitter=.4, size=1, color='k')

    ax.plot(2 * bin_centers,  label='2x', color='#f0c220')
    ax.plot(3 * bin_centers, label='3x', color='#7a6952')
    ax.plot(4 * bin_centers, label='4x', color='#c72321')
    ax.plot(6 * bin_centers, label='6x', color='#fbd7a9')

    colors = '#246b71', '#6a9395', '#84bcbf', '#9bdade'
    for q, color in zip((0.05, 0.1, 0.2, 0.3), colors):
        ax.plot(pd.Series(min_distances_not_nan).groupby(bin_idcs).quantile(q=q).values,
                label=f"{int(q * 100)}\\%% quantile", color=color)

    ax.set_ylim(0, 500)
    ax.set_xlim(0, 11)

    plt.ylabel('Distance to closest turbine [m]')
    plt.xlabel('Rotor diameter [m]')

    plt.legend()

    plt.grid()

    return fig


def plot_power_curves():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    x = np.linspace(0, 40, num=100)
    colors = '#c72321', '#7a6952', '#0d8085', '#f0c220'
    for turbine_model, color in zip((ge15_77,) + new_turbine_models(), colors):
        linestyle = 'dashed' if turbine_model == ge15_77 else '-'
        ax.plot(x, turbine_model.power_curve(x), label=turbine_model.name, color=color,
                linestyle=linestyle)
    plt.legend()
    plt.ylabel('Power generation [kW]')
    plt.xlabel('Wind speed [m/s]')

    plt.grid()

    return fig


def plot_wind_rose(data1, data2=None, args=None, kwargs=None):
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
    args : iterable
        passed directly to ax.plot()
    kwargs : dict
        passed directly to ax.plot()


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
        directions = np.linspace(-np.pi, np.pi, num=len(values))
    elif isinstance(data2, str):
        raise ValueError(f"invalid type str for data2: {data2}")
    else:
        directions = data1
        values = data2

    # close circle if first point and last are not the same, might plot points double
    directions = np.append(directions, directions[0])
    values = np.append(values, values[0:1], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, subplot_kw=dict(polar=True))

    # TODO there might be a 180° error in here, it is calibrated to ERA5 data and wind roses
    #  it is not entirely clear which direction it should go, would does North mean? North wind?
    ax.plot(-directions - np.pi/2., values, *args, **kwargs)
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')

    return fig
