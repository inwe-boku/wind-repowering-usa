from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from wind_repower_usa.config import EXTERNAL_DIR

Turbine = namedtuple('Turbine', ('name',
                                 'file_name',
                                 'power_curve',
                                 'capacity_mw',
                                 'rotor_diameter_m',
                                 'hub_height_m'))


def new_turbine_models():
    """Return all turbine models used for repowering."""
    return se_34m114, e138ep3, se_42m140, e126


def turbine_from_nrel_sam(wind_turbine_model, file_name, name=None):
    """Pass one row of the CSV as `wind_turbine_model` and a machine-readable name as `file_name`"""
    wind_speeds = [float(x) for x in wind_turbine_model['Wind Speed Array'].split('|')]
    generation_kw = [float(x) for x in wind_turbine_model['Power Curve Array'].split('|')]

    assert generation_kw[-1] == 0., "no cut-off in power curve, need to be fixed manually"

    wind_speeds += [70.]
    generation_kw += [0.]

    if wind_speeds[0] > 0.:
        wind_speeds = [0.] + wind_speeds
        generation_kw = [0.] + generation_kw

    turbine = Turbine(
        name=name or wind_turbine_model['Name'],
        file_name=file_name,
        power_curve=interp1d(wind_speeds, generation_kw),
        capacity_mw=wind_turbine_model['KW Rating'] * 1e-3,
        rotor_diameter_m=wind_turbine_model['Rotor Diameter'],
        hub_height_m=None,
    )
    return turbine


wind_turbine_models = pd.read_csv(EXTERNAL_DIR / 'nrel-sam-powercurves' /
                                  'nrel-sam-wind-turbines.csv', skiprows=[1, 2])


se_34m114 = turbine_from_nrel_sam(wind_turbine_models.iloc[267], 'se_34m114',
                                  name='Senvion 3.4M114 (3.4MW, 114m)')
vestas_v42_600 = turbine_from_nrel_sam(wind_turbine_models.iloc[145], 'vestas_v42_600')
northwind100 = turbine_from_nrel_sam(wind_turbine_models.iloc[118], 'northwind100')
e44 = turbine_from_nrel_sam(wind_turbine_models.iloc[165], 'e44')


def power_curve_ge15_77():
    """Power curve for GE1.5-77
    https://www.ge.com/in/wind-energy/1.5-MW-wind-turbine

    Hub height: 65 / 80m

    https://www.nrel.gov/docs/fy15osti/63684.pdf  page 21
    https://geosci.uchicago.edu/~moyer/GEOS24705/Readings/GEA14954C15-MW-Broch.pdf
    """
    wind_speeds = np.hstack((np.arange(0, 27, step=1.5), [29., 70]))
    generation_kw = [0., 0., 0., 70., 210., 520., 930., 1280., 1470, 1500.] + [1500.] * 8 + [0., 0.]
    return interp1d(wind_speeds, generation_kw)


ge15_77 = Turbine(
    name='GE1.5-77 (1.5MW, 77m)',
    file_name='ge15_77',
    power_curve=power_curve_ge15_77(),
    capacity_mw=1.5,
    rotor_diameter_m=77.,
    hub_height_m=None,
)


def power_curve_e138ep3():
    """Power curve for Enercon E-138 EP3

    https://www.enercon.de/en/products/ep-3/e-138-ep3/

    Rated power 	3,500 kW
    Rotor diameter 	138,6 m
    Hub height in meter 	81 / 111 / 131 / 160
    """
    wind_speeds = np.hstack((np.linspace(0, 25, num=26), [26, 70]))
    generation_kw = [0., 0., 0., 30., 200., 490., 950., 1400.,  2050.,  2550., 3100., 3400.,
                     3480, 3500., 3500., 3500., 3500., 3500., 3500., 3500., 3500., 3480., 3410.,
                     3300, 3200., 3000., 0., 0.]

    return interp1d(wind_speeds, generation_kw)


e138ep3 = Turbine(
    name='Enercon E-138 EP3 (3.5MW, 138m)',
    file_name='e138ep3',
    power_curve=power_curve_e138ep3(),
    capacity_mw=3.5,
    rotor_diameter_m=138.6,
    hub_height_m=None,
)


def power_curve_se_42m140():
    """Power curve for Senvion 4.2M140
    Extracted by hand from offial spec-sheet at:
    https://www.senvion.com/global/en/products-services/wind-turbines/4xm/
    """
    wind_speeds = np.hstack(([0., 1., 2., 3.], np.arange(4, 27, step=2), [27., 40.]))
    generation_kw = [0., 0., 0., 0., 300., 1050., 2700., 4000., 4200., 4200., 4200., 4200.,
                     4200., 4000., 2500., 650., 0., 0.]
    return interp1d(wind_speeds, generation_kw)


se_42m140 = Turbine(
    name='Senvion 4.2M140 (4.2MW, 140m)',
    file_name='se_42m140',
    power_curve=power_curve_se_42m140(),
    capacity_mw=4.2,
    rotor_diameter_m=140.,
    hub_height_m=None,
)


def power_curve_e126():
    """Power curve for Enercon E-126 (7.580MW Onshore)
    Extracted by hand from official datasheet at:
    https://www.enercon.de/produkte/ep-8/e-126/

    See also:
    https://www.enercon.de/fileadmin/Redakteur/Medien-Portal/broschueren/pdf/en/ENERCON_Produkt_en_06_2015.pdf

    Cut-out: 28-34m/s

    """
    wind_speeds = np.hstack((np.arange(0, 29, step=1), [34., 70.]))
    generation_kw = [0., 0., 0., 100., 200., 400., 800., 1200., 2000., 2800., 3700., 4900., 5700.,
                     6500., 7000., 7300., 7580., 7580., 7580., 7580., 7580., 7580., 7580.,
                     7580., 7580., 7580., 7580., 7580., 7580., 0., 0.]
    return interp1d(wind_speeds, generation_kw)


e126 = Turbine(
    name='Enercon E-126 (7.58MW, 127m)',
    file_name='e126',
    power_curve=power_curve_e126(),
    capacity_mw=7.58,
    rotor_diameter_m=127.,
    hub_height_m=135,
)
