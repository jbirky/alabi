import numpy as np
from scipy.optimize import rosen

__all__ = ["rosenbrock",
           "gaussian_shells",
           "eggbox"]


# ================================
# Rosenbrock function (2D)
# ================================

def rosenbrock_fn(x):
    return -rosen(x)/100

rosenbrock_bounds = [(-5,5), (-5,5)]

rosenbrock = {"fn": rosenbrock_fn,
              "bounds": rosenbrock_bounds}


# ================================
# Gaussian shells (2D)
# ================================

def logcirc(theta, c):
    r = 2.  # radius
    w = 0.1  # width
    c1 = np.array([-3.5, 0.])  # center of shell 1
    c2 = np.array([3.5, 0.])  # center of shell 2
    const = math.log(1. / math.sqrt(2. * math.pi * w**2))  # normalization constant
    d = np.sqrt(np.sum((theta - c)**2, axis=-1))  # |theta - c|
    return const - (d - r)**2 / (2. * w**2)

def gaussian_shells_fn(theta):
    return np.logaddexp(logcirc(theta, c1), logcirc(theta, c2))

gaussian_shells_bounds = [(-5,5), (-5,5)]

gaussian_shells = {"fn": gaussian_shells_fn,
                   "bounds": gaussian_shells_bounds}


# ================================
# Eggbox (2D)
# ================================

def eggbox_fn(x):
    tmax = 5.0 * np.pi
    t = 2.0 * tmax * x - tmax
    return (2.0 + np.cos(t[0] / 2.0) * np.cos(t[1] / 2.0)) ** 5.0

eggbox_bounds = [(0,1), (0,1)]

eggbox = {"fn": eggbox_fn,
          "bounds": eggbox_bounds}