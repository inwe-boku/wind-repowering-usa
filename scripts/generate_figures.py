import xarray as xr
import matplotlib.pyplot as plt

from wind_repower_usa.calculations import calc_mean_wind_speed
from wind_repower_usa.config import DISTANCE_FACTORS, FIGURES_DIR, INTERIM_DIR
from wind_repower_usa.load_data import load_turbines, load_cluster_per_location
from wind_repower_usa.load_data import load_repower_potential
from wind_repower_usa.load_data import load_optimal_locations
from wind_repower_usa.load_data import load_generated_energy_gwh
from wind_repower_usa.load_data import load_prevail_wind_direction
from wind_repower_usa.load_data import load_distance_factors
from wind_repower_usa.logging_config import setup_logging
from wind_repower_usa.turbine_models import new_turbine_models, e138ep3
from wind_repower_usa.visualize import plot_repower_potential
from wind_repower_usa.visualize import plot_mean_wind_speed_and_turbines
from wind_repower_usa.visualize import plot_optimized_cluster
from wind_repower_usa.visualize import plot_simulated_generated_energy
from wind_repower_usa.visualize import plot_history_turbines
from wind_repower_usa.visualize import plot_min_distances
from wind_repower_usa.visualize import plot_power_curves


# https://matplotlib.org/users/usetex.html
# https://matplotlib.org/gallery/userdemo/pgf_texsystem.html
# TODO this is probably the failed try to make matplotlib and latex fonts equal
# plt.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "pgf.preamble": [
#          r"\usepackage[T1]{fontenc}",
#          r"\usepackage{cmbright}",
#          ]
# })
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['mathtext.fontset'] = 'stixsans'

plt.rcParams["font.size"] = "14"


def savefig_history_turbines():
    plot_history_turbines()
    plt.savefig(FIGURES_DIR / 'history_turbines.pdf', bbox_inches='tight')


def savefig_power_curves():
    plot_power_curves()
    plt.savefig(FIGURES_DIR / 'power_curves.pdf', bbox_inches='tight')


def savefig_repower_potential():
    repower_potentials = []
    for distance_factor in DISTANCE_FACTORS:
        for turbine_model_new in ('mixed',) + new_turbine_models():
            repower_potentials.append(load_repower_potential(turbine_model_new, distance_factor))

    plot_repower_potential(*repower_potentials, variable='power_generation')
    plt.savefig(FIGURES_DIR / 'repower_potential_power_generation.pdf', bbox_inches='tight')

    plot_repower_potential(*repower_potentials, variable='num_turbines')
    plt.savefig(FIGURES_DIR / 'repower_potential_num_turbines.pdf', bbox_inches='tight')


def savefig_repower_potential_direction():
    repower_potentials = []
    for turbine_model_new in ('mixed',) + new_turbine_models():
        repower_potentials.append(load_repower_potential(turbine_model_new, distance_factor=None))

    plot_repower_potential(*repower_potentials, variable='power_generation')
    plt.savefig(FIGURES_DIR / 'repower_potential-direction-dependent_power_generation.pdf',
                bbox_inches='tight')

    plot_repower_potential(*repower_potentials, variable='num_turbines')
    plt.savefig(FIGURES_DIR / 'repower_potential-direction-dependent_num_turbines.pdf',
                bbox_inches='tight')


def savefig_mean_wind_speed_and_turbines(turbines):
    wind_speed_mean = calc_mean_wind_speed(years=range(2010, 2019),
                                           sample_size=200)
    plot_mean_wind_speed_and_turbines(wind_speed_mean, turbines)
    plt.savefig(FIGURES_DIR / 'mean_wind_speed_and_turbines.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'mean_wind_speed_and_turbines.png', bbox_inches='tight', dpi=150)


def savefig_optimized_cluster(turbines):
    turbine_model = e138ep3
    is_optimal_location = load_optimal_locations(turbine_model, None)
    cluster_per_location = load_cluster_per_location(None)

    prevail_wind_direction = load_prevail_wind_direction()
    distance_factors = load_distance_factors()

    fig = plot_optimized_cluster(turbines, cluster_per_location, is_optimal_location,
                                 turbine_model, distance_factors, prevail_wind_direction)
    fig.savefig(FIGURES_DIR / 'optimized_cluster.pdf', bbox_inches='tight')


def savefig_simulated_energy_time_series():
    generated_energy_gwh = load_generated_energy_gwh()
    simulated_energy_gwh = xr.open_dataarray(INTERIM_DIR / 'simulated_energy_timeseries' /
                                             'simulated_energy_timeseries_ge15_77_gwh.nc')
    simulated_energy_gwh = simulated_energy_gwh.sel(time=slice(generated_energy_gwh.time.min(),
                                                               None))

    plot_simulated_generated_energy(simulated_energy_gwh)
    plt.savefig(FIGURES_DIR / 'simulated-energy_time-series.pdf', bbox_inches='tight')


def savefig_min_distances(turbines):
    min_distances = xr.open_dataarray(INTERIM_DIR / 'min_distances' / 'min_distances.nc')
    plot_min_distances(turbines, min_distances)
    plt.savefig(FIGURES_DIR / 'min_distances_between_turbines.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'min_distances_between_turbines.png', bbox_inches='tight', dpi=150)


def save_figures():
    turbines = load_turbines()

    savefig_history_turbines()
    savefig_power_curves()
    savefig_repower_potential()
    savefig_repower_potential_direction()
    savefig_mean_wind_speed_and_turbines(turbines)
    savefig_optimized_cluster(turbines)
    savefig_simulated_energy_time_series()
    savefig_min_distances(turbines)


if __name__ == '__main__':
    setup_logging()
    save_figures()
