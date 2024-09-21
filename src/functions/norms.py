import numpy as np
from scipy.stats import median_abs_deviation
from scipy.spatial import distance

def median_absolute_deviation(X, a, b):
    return (abs(a - b))/median_abs_deviation(X)

def manhattan_distance(a, b):
    dist = distance.cityblock(a, b)
    return dist

def square_difference(a, b):
    return (a - b)**2

def fidelity_error(x, y, unit):
    x, y, unit = np.array(x), np.array(y), np.array(unit)
    b_pert = np.linalg.norm(x - unit)
    est_b_pert = np.linalg.norm(y - unit)
    
    return abs(b_pert - est_b_pert)

def proximity_score(x, y):
    """
    Compute the proximity score between the original instance x and the counterfactual instance y.
    """
    print(type(x))

    
    cat_diff = np.sum(y != x)
    
    return -cat_diff