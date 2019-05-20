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
        age in "years before 2016", maybe actually: (date of availability)-2015

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
    # the numbers from the paper can be reproduced with age = 0
    for i in range(20):
        print(i, turbine_cost(75, 3000000, 90, i))

    # Test turbine 1 (vintage) from Rinne et al:
    # Year: 2002, High winds, V90-3.0 MW, Height: 75, Price: 878 Euro/kW
    print("vintage:", turbine_cost(75, 3000000, 90, 0))

    # Test turbine 2 (new) from Rinne et al:
    # Year: 2015, High winds, V117-3.45 MW, Height: 125, Price:  1,448 Euro/kW
    print("new:", turbine_cost(125, 3450000, 117, 0))
