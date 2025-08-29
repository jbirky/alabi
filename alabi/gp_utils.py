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
import copy
import warnings

__all__ = ["default_hyper_prior", 
           "configure_gp", 
           "optimize_gp",
           "grad_gp_mean_prediction",
           "grad_gp_var_prediction"]


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


def grad_gp_mean_prediction(xs, gp):
    """
    Compute the gradient of the GP mean prediction with respect to the input
    locations.

    :param xs: (*array-like, required*)
        Input locations to evaluate the gradient at.

    :param gp: (*george.GP, required*)
        The computed Gaussian process object.

    :returns: (*array*)
        The gradient of the GP mean prediction at the input locations.
    """
    xs = np.asarray(xs).reshape(1, -1)
    grad_ks = gp.kernel.get_gradient(xs, gp._x)[0]
    
    return np.dot(grad_ks.T, gp._alpha)


def grad_gp_var_prediction(xs, gp):
    """
    Compute the gradient of the GP variance prediction with respect to the input
    locations.

    :param xs: (*array-like, required*)
        Input locations to evaluate the gradient at.

    :param gp: (*george.GP, required*)
        The computed Gaussian process object.

    :returns: (*array*)
        The gradient of the GP variance prediction at the input locations.
    """
    xs = np.array(xs).reshape(1, -1)
    grad_ks = gp.kernel.get_gradient(xs, gp._x)[0]
    grad_kss = gp.kernel.get_gradient(xs, xs).flatten() 
    ks = gp.kernel.get_value(xs, gp._x).flatten()
    Kinv = gp.solver.get_inverse()
    
    return grad_kss - 2 * np.dot(grad_ks.T, np.dot(Kinv, ks))


def configure_gp(theta, y, kernel, 
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
                method="l-bfgs-b", options=None, bounds=None):

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Collapse arrays if 1D
    theta = theta.squeeze()
    y = y.squeeze()

    # initial hyperparameters
    init_hp = gp.get_parameter_vector()
        
    valid_methods = ["newton-cg", "l-bfgs-b", "powell"]
    if method not in valid_methods:
        print(f"Warning: {method} not in valid methods {valid_methods}. Using 'l-bfgs-b' optimizer instead.")
        method = "l-bfgs-b"

    # Minimize GP nll, save result, evaluate marginal likelihood
    if method in ["newton-cg", "l-bfgs-b"]:
        jac = _grad_nll
    else:
        jac = None
        
    # Set improved default options for faster convergence
    if options is None:
        default_options = {
            'newton-cg': {'maxiter': 50, 'xtol': 1e-4, 'gtol': 1e-4},
            'l-bfgs-b': {'maxiter': 50, 'gtol': 1e-4, 'ftol': 1e-6},
            'powell': {'maxiter': 100, 'xtol': 1e-4, 'ftol': 1e-6},
        }
        options = default_options.get(method.lower(), {})
    
    nopt = p0.shape[0] if p0.ndim > 1 else 1
    if nopt > 1:
        # Run the optimization routine nopt times
        res = []
        mll = []
        
        for ii, x0 in enumerate(p0):
            try:
                result = minimize(_nll, x0, args=(gp, y, gp_hyper_prior), method=method,
                                jac=jac, bounds=bounds, options=options)
                
                if result.success and np.isfinite(gp_hyper_prior(result.x)):
                    # Compute marginal log likelihood for this set of kernel hyperparameters
                    test_gp = copy.copy(gp)
                    test_gp.set_parameter_vector(result.x)
                    test_gp.recompute()
                    current_mll = test_gp.log_likelihood(y, quiet=True)
                    
                    res.append(result.x)
                    mll.append(current_mll)
                            
                else:
                    print(f"\nWarning: GP hyperparameter optimization restart {ii} failed. Solution failed prior bounds.\n")
                    res.append(init_hp)
                    mll.append(-np.inf)
                    
            except Exception as e:
                print(f"\nWarning: GP hyperparameter optimization restart {ii} failed with error: {e}\n")
                res.append(init_hp)
                mll.append(-np.inf)

        # Pick result with largest marginal log likelihood
        if len(mll) > 0 and max(mll) > -np.inf:
            ind = np.argmax(mll)
            try:
                gp.set_parameter_vector(res[ind])   
                gp.recompute()
            except:
                print("\nWarning: Failed to set best hyperparameters. Using initial values.\n")
                gp.set_parameter_vector(init_hp)
                gp.recompute()
        else:
            print("\nWarning: All hyperparameter optimizations failed. Using initial values.\n")
            gp.set_parameter_vector(init_hp)
            
    else:
        # Optimize GP hyperparameters by maximizing marginal log_likelihood
        res = minimize(_nll, p0, args=(gp, y, gp_hyper_prior), method=method,
                       jac=jac, bounds=bounds, options=options)

        if not res.success:
            print("\nWarning: GP hyperparameter optimization failed.\n")
            return None

        # Set the optimized hyperparameters
        try:
            gp.set_parameter_vector(res.x)
            gp.recompute()
        except:
            print("\nWarning: GP hyperparameter optimization failed. Cannot recompute gp.\n")
            return None

    return gp