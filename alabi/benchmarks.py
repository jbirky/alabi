"""
:py:mod:`benchmarks.py` 
-------------------------------------
"""

import numpy as np
from scipy.optimize import rosen
import math

__all__ = ["test1d",
           "rosenbrock",
           "gaussian_shells",
           "eggbox", 
           "multimodal"]


# ================================
# 1D test function (1D)
# ================================

def test1d_fn(theta):
    """
    Simple 1D test Bayesian optimization function adapted from
    https://krasserm.github.io/2018/03/21/bayesian-optimization/
    """

    theta = np.asarray(theta)
    return -np.sin(3*theta) - theta**2 + 0.7*theta

test1d_bounds = [(-2,1)]

test1d = {"fn": test1d_fn,
          "bounds": test1d_bounds}


# ================================
# Rosenbrock function (2D)
# ================================

def rosenbrock_fn(x):
    return -rosen(x)/100.0

rosenbrock_bounds = [(-5,5), (-5,5)]

rosenbrock = {"fn": rosenbrock_fn,
              "bounds": rosenbrock_bounds}


# ================================
# Gaussian shells (2D)
# ================================

def logcirc(theta, c):
    r = 2.  # radius
    w = 0.1  # width
    const = math.log(1. / math.sqrt(2. * math.pi * w**2))  # normalization constant
    d = np.sqrt(np.sum((theta - c)**2, axis=-1))  # |theta - c|
    return const - (d - r)**2 / (2. * w**2)

def gaussian_shells_fn(theta):
    c1 = np.array([-3.5, 0.])  # center of shell 1
    c2 = np.array([3.5, 0.])  # center of shell 2
    return np.logaddexp(logcirc(theta, c1), logcirc(theta, c2))

gaussian_shells_bounds = [(-6,6), (-6,6)]

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


# ================================
# Multimodal function (2D)
# ================================

def multimodal_fn(x):
    "https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html"
    return -(np.sin(x[0]) ** 10 + np.cos(10 + x[1] * x[0]) * np.cos(x[0]))

multimodal_bounds = [(0,5), (0,5)]

multimodal = {"fn": multimodal_fn,
              "bounds": multimodal_bounds}