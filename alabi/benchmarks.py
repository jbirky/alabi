"""
:py:mod:`benchmarks.py` 
-------------------------------------
"""

import numpy as np
from scipy.optimize import rosen
from scipy.interpolate import interp2d
from scipy.stats import multivariate_normal
import math

__all__ = ["test1d",
           "rosenbrock",
           "gaussian_shells",
           "eggbox", 
           "gaussian_2d",
           "multimodal",
           "logo",
           "random_gaussian_covariance",
           "multimodal_gaussian_nd"]


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
# An N-dimensional Rosenbrock function 
# ================================

def rosenbrock_nd(x, a, b):
    """
    ND Rosenbrock function from Pagani et al. (2020): https://arxiv.org/pdf/1903.09556
    
    :param x: input vector of shape (n_samples, ndim) or (ndim,)
    :param a: scalar parameter
    :param b: matrix of parameters
    """
    n1, n2 = b.shape
    ndim = (n1 - 1)*n2 + 1
    
    # Handle both 1D and 2D input
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Vectorized computation for multiple samples
    log_like = - a*(x[:, 0] - 1)**2
    cnorm = np.sqrt(a / np.pi) * np.pi**ndim
    
    # Compute diff term for all samples: shape (n_samples, n1-2)
    diff_term = (x[:, 2:n1] - x[:, 1:n1-1]**2)**2
    # Sum b coefficients along n2 dimension: shape (n1-2,)
    b_sum_per_col = b[:, 2:].sum(axis=0)
    # Multiply and sum over dimensions
    log_like -= (diff_term * b_sum_per_col).sum(axis=1)
    
    cnorm *= np.sqrt(np.prod(b[:, 2:]))
    log_like -= np.log(cnorm)
    
    if squeeze_output:
        return log_like[0]
    return log_like


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
    theta = np.asarray(theta).flatten()
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
    x = np.asarray(x).flatten()
    tmax = 5.0 * np.pi
    t = 2.0 * tmax * x - tmax
    return -(2.0 + np.cos(t[0] / 2.0) * np.cos(t[1] / 2.0)) ** 5.0

eggbox_bounds = [(0,1), (0,1)]

eggbox = {"fn": eggbox_fn,
          "bounds": eggbox_bounds}


# ================================
# Multimodal function (2D)
# ================================

def multimodal_fn(x):
    "https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html"
    x = np.asarray(x).flatten()
    return -(np.sin(x[0]) ** 10 + np.cos(10 + x[1] * x[0]) * np.cos(x[0]))

multimodal_bounds = [(0,5), (0,5)]

multimodal = {"fn": multimodal_fn,
              "bounds": multimodal_bounds}


# ================================
# Logo (2D)
# ================================

def logo_fn(theta):
    data = np.loadtxt('../benchmark/logo.txt')

    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = data[::-1,::]

    f = interp2d(x, y, Z, kind='linear')

    return f(theta[0], theta[1])[0]

logo_bounds = [(0,355), (0,132)]

logo = {"fn": logo_fn,
        "bounds": logo_bounds}


# ================================
# Gaussian  (2D)
# ================================


def gaussian_2d_fn(theta):
    """
    2D Gaussian function with mean at the center and covariance matrix as identity.
    """
    theta = np.asarray(theta).flatten()
    mean = np.array([0.5, 0.5])
    cov = np.array([[0.1, 0.0], [0.0, 0.1]])
    return multivariate_normal.logpdf(theta, mean=mean, cov=cov)

gaussian_2d_bounds = [(0,1), (0,1)]
gaussian_2d = {"fn": gaussian_2d_fn,
               "bounds": gaussian_2d_bounds}


# ================================
# N-dimensional gaussian (ND)
# ================================

def random_gaussian_covariance(n_dims):
    """
    Generate a random positive definite covariance matrix.
    """
    eigenvals = np.random.exponential(scale=1.0, size=n_dims)
    # Generate random orthogonal matrix (eigenvectors)
    Q = np.random.randn(n_dims, n_dims)
    Q, _ = np.linalg.qr(Q)  # QR decomposition gives orthogonal Q
    
    # Construct covariance matrix: C = Q * Λ * Q^T
    cov = Q @ np.diag(eigenvals) @ Q.T
    return cov


def multimodal_gaussian_nd(x, means, covs, amps):
    
    nmodes = len(means)
    log_prob = np.array([ amps[ii] * multivariate_normal.logpdf(x, mean=means[ii], cov=covs[ii]) for ii in range(nmodes) ])
    prob = np.sum(np.exp(log_prob), axis=0)

    return np.exp(prob)