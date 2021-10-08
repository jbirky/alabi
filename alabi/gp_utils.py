"""
:py:mod:`gp_utils.py` 
-------------------------------------------------

Gaussian process utility functions for initializing GPs and optimizing their
hyperparameters.

"""

import numpy as np
import george
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from functools import partial

__all__ = ["default_hyper_prior", 
           "configure_gp", 
           "optimize_gp"]


def default_hyper_prior(p, hp_rng=20, mu=None, sigma=None, sigma_level=3):
    """
    Default prior function for GP hyperparameters. This prior also keeps the
    hyperparameters within a reasonable huge range ``[-20, 20]``. Note that george
    operates on the *log* hyperparameters, except for the mean function.

    :param p: (*array, required*) 
        Array of GP hyperparameters

    :returns lnprior: (*float*) 
        log prior value 
    """

    # Restrict range of hyperparameters (ignoring mean term)
    if np.any(np.fabs(p)[1:] > hp_rng):
        return -np.inf

    # Restrict mean hyperparameter to +/- sigma_level * sigma
    if (mu is not None) and (sigma is not None):
        if (p[0] < mu - sigma_level*sigma) or (p[0] > mu + sigma_level*sigma):
            return -np.inf

    return 0.0


def _nll(p, gp, y, prior_fn=None):
    """
    Given parameters and data, compute the negative log likelihood of the data
    under the george Gaussian process.

    Parameters
    ----------
    p : array
        GP hyperparameters
    gp : george.GP
    y : array
        data to condition GP on
    prior_fn : callable
        Prior function for the GP hyperparameters, p

    Returns
    -------
    nll : float
        negative log-likelihood of y under gp
    """

    # Apply priors on GP hyperparameters
    if prior_fn is not None:
        if not np.isfinite(prior_fn(p)):
            return np.inf

    # Catch singular matrices
    try:
        gp.set_parameter_vector(p)
    except np.linalg.LinAlgError:
        return np.inf

    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else np.inf


def _grad_nll(p, gp, y, prior_fn=None):
    """
    Given parameters and data, compute the gradient of the negative log
    likelihood of the data under the george Gaussian process.

    Parameters
    ----------
    p : array
        GP hyperparameters
    gp : george.GP
    y : array
        data to condition GP on
    prior_fn : callable
        Prior function for the GP hyperparameters, p

    Returns
    -------
    gnll : float
        gradient of the negative log-likelihood of y under gp
    """

    # Apply priors on GP hyperparameters
    if prior_fn is not None:
        if not np.isfinite(prior_fn(p)):
            return np.inf

    # Negative gradient of log likelihood
    return -gp.grad_log_likelihood(y, quiet=True)


def configure_gp(theta, y, kernel, gp_hyper_prior,  
                 fit_amp=True, fit_mean=True, fit_white_noise=False,
                 white_noise=-12, hyperparameters=None):

    if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(y)):
        print("theta, y:", theta, y)
        raise ValueError("All theta and y values must be finite!")

    if fit_amp == True:
        kernel *= np.var(y)

    gp = george.GP(kernel=kernel, fit_mean=fit_mean, mean=np.median(y),
                   white_noise=white_noise, fit_white_noise=fit_white_noise)

    if hyperparameters is not None:
        gp.set_parameter_vector(hyperparameters)

    try:
        gp.compute(theta)
    except:
        gp = None

    return gp


def optimize_gp(gp, theta, y, gp_hyper_prior, p0,
                nopt=3, method="powell", options=None):
    
    # Collapse arrays if 1D
    theta = theta.squeeze()
    y = y.squeeze()
    
    # Minimize GP nll, save result, evaluate marginal likelihood
    if method not in ["nelder-mead", "powell", "cg"]:
        jac = _grad_nll
    else:
        jac = None
    
    # Run the optimization routine nopt times
    res = []
    mll = []

    # Optimize GP hyperparameters by maximizing marginal log_likelihood
    for ii, x0 in enumerate(p0):

        resii = minimize(_nll, x0, args=(gp, y, gp_hyper_prior), method=method,
                         jac=jac, bounds=None, options=options)["x"]
        res.append(resii)
        try:
            # Update the kernel with solution for computing marginal loglike
            gp.set_parameter_vector(resii)
            gp.recompute()

            # Compute marginal log likelihood for this set of kernel hyperparameters
            mll.append(gp.log_likelihood(y, quiet=True))
        except:
            # solution not valid
            mll.append(-np.inf)

    # Pick result with largest marginal log likelihood
    ind = np.argmax(mll)

    # if hyperparameters allowed by prior
    if np.isfinite(gp_hyper_prior(res[ind])):
        # Update gp
        gp.set_parameter_vector(res[ind])
        try:
            gp.recompute()
        except:
            print("\nWarning: GP hyperparameter optimization failed. Cannot recompute gp.\n")
            gp = None
    else:
        print("\nWarning: GP hyperparameter optimization failed. Solution failed prior bounds.\n")
        gp = None

    return gp