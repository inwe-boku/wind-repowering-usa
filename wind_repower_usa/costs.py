import math
import numpy as np


def turbine_cost(height, rated_capacity, diameter, age):
    """Implements Eq (8) from Rinne et al 2018

    Parameters
    ----------
    height
        in m
    rated_capacity
        in W
    diameter
        in m
    age
        age in "years before 2016" i think in reality its (date of availability)-2015

    Returns
    -------
        Cost in Euro 2016

    """
    beta1 = 620
    beta2 = -1.68
    beta3 = 182
    C = -1005
    radius = diameter*0.5

    specific_power = rated_capacity/(math.pi*radius**2)
    cost = beta1 * np.log(height) + beta2 * specific_power + beta3 * math.sqrt(age) + C

    return cost


if __name__ == '__main__':
    print(turbine_cost(1, 1, 1, 1))

    # Testturbine 1 from Rinne et al:
    # Year 2002 High winds V90-3.0 MW Height: 75 Price: 878 euro/kW
    # Whoaaaa this is completely crazy!
    # the numbers from the paper are for vintage setting age=0!
    # and new age=1!
    for i in range(20):
        print(i)
        print(turbine_cost(75, 3000000, 90, i))

    print(turbine_cost(75, 3000000, 90, 0))

    # Testturbine 2 from Rinne et al:
    # Year 2015 High winds V117-3.45 MW Height 125 Price:  1,448 euro/kW
    print(turbine_cost(125, 3450000, 117, 1))
