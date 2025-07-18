# -*- coding: utf-8 -*-
"""
:py:mod:`utility.py`
----------------------------------------

Utility functions in terms of usefulness, e.g. minimizing GP utility functions
or computing KL divergences, and the GP utility functions, e.g. the bape utility.
"""

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm
from skopt.space import Space
from skopt.space.space import Real, Categorical
from skopt.sampler import Sobol, Lhs, Halton, Hammersly, Grid
import multiprocessing as mp
from functools import partial
import warnings
import time
import tqdm

__all__ = ["agp_utility", 
           "bape_utility", 
           "jones_utility",
           "assign_utility",
           "minimize_objective", 
           "prior_sampler", 
           "prior_sampler_normal",
           "eval_fn", 
           "lnprior_uniform",
           "prior_transform_uniform",
           "lnprior_normal",
           "prior_transform_normal",
           "flatten_array"]


def flatten_array(arr, method='flatten'):
    """
    Flatten a numpy array using different methods.
    
    :param arr: (*array, required*)
        Input numpy array to flatten.
    
    :param method: (*str, optional*)
        Method to use for flattening. Defaults to 'flatten'.
        Options:
            'flatten' - Returns a copy of the array collapsed into 1D
            'ravel' - Returns a contiguous flattened array (view if possible)
            'flat' - Returns a flat iterator over the array
            'reshape' - Reshapes the array to 1D using reshape(-1)
    
    :returns flattened_array: (*array*)
        Flattened numpy array.
    
    :example:
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
        >>> flatten_array(arr)
        array([1, 2, 3, 4, 5, 6])
        >>> flatten_array(arr, method='ravel')
        array([1, 2, 3, 4, 5, 6])
    """
    
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    
    if method == 'flatten':
        return arr.flatten()
    elif method == 'ravel':
        return arr.ravel()
    elif method == 'flat':
        return np.array([x for x in arr.flat])
    elif method == 'reshape':
        return arr.reshape(-1)
    else:
        err_msg = f"Method '{method}' not implemented. "
        err_msg += "Valid options for 'method' are: "
        err_msg += "flatten, ravel, flat, reshape."
        raise ValueError(err_msg)


#===========================================================
# Define sampling functions
#===========================================================

def prior_sampler(bounds=None, nsample=1, sampler='uniform'):
    """
    Hypercube sampling function which draws ``nsample`` within given ``bounds``. 
    Wrapper around ``scikit-optimize`` sampling methods. For more info see:
    https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html

    :param bounds: (*array, required*)
        Array of ``(min,max)`` bounds for each dimension of the prior.

    :param nsample: (*int, optional*)
        Defaults to ``nsample=1``.

    :param sampler: (*str, optional*)
        Defaults to ``'uniform'``. Options:
            ``'uniform'``,
            ``'sobol'``,
            ``'lhs'``,
            ``'halton'``,
            ``'hammersly'``,
            ``'grid'``
    """

    ndim = len(bounds)
        
    # space_bounds = [Categorical(categories=(np.float64(bounds[ii][0]), np.float64(bounds[ii][1]))) for ii in range(ndim)]
    # space_bounds = [Real(bounds[ii][0], bounds[ii][1], dtype='float') for ii in range(ndim)]
    space = Space(bounds)
    
    if sampler == 'uniform':
        samples = space.rvs(nsample)

    elif sampler == 'sobol':
        sobol = Sobol()
        samples = sobol.generate(space.dimensions, nsample)

    elif sampler == 'lhs':
        lhs = Lhs(lhs_type="classic", criterion=None)
        samples = lhs.generate(space.dimensions, nsample)

    elif sampler == 'halton':
        halton = Halton()
        samples = halton.generate(space.dimensions, nsample)

    elif sampler == 'hammersly':
        hammersly = Hammersly()
        samples = hammersly.generate(space.dimensions, nsample)

    elif sampler == 'grid':
        grid = Grid(border="include", use_full_layout=False)
        samples = grid.generate(space.dimensions, nsample)
            
    else:
        err_msg = f"Sampler method '{sampler}' not implemented. "
        err_msg += f"Valid options for 'sampler' are: "
        err_msg += f"uniform, sobol, lhs, halton, hammersly, grid."
        raise ValueError(err_msg)

    return np.array(samples) 


def prior_sampler_normal(prior_data, bounds, nsample=1):

    ndim = len(bounds)

    rvs = np.zeros((ndim, nsample))
    for ii in range(ndim):
        if prior_data[ii][0] is not None:
            lb = (bounds[ii][0] - prior_data[ii][0]) / prior_data[ii][1]
            ub = (bounds[ii][1] - prior_data[ii][0]) / prior_data[ii][1]
            rvs[ii] = truncnorm.rvs(lb, ub, loc=prior_data[ii][0], scale=prior_data[ii][1], size=nsample)
        else:
            rvs[ii] = np.random.uniform(low=bounds[ii][0], high=bounds[ii][1], size=nsample)
    
    return rvs.T

    
def eval_fn(fn, theta, ncore=mp.cpu_count()):

    t0 = time.time()

    if ncore <= 1:
        y = np.zeros(theta.shape[0])
        for ii, tt in tqdm.tqdm(enumerate(theta)):
            y[ii] = fn(tt)
    else:
        with mp.Pool(ncore) as p:
            y = np.array(p.map(fn, theta))

    tf = time.time()
    print(f"Computed {len(theta)} function evaluations: {np.round(tf - t0)}s \n")

    try:
        return y.squeeze()
    except:
        return y


def lnprior_uniform(x, bounds):

    ndim = len(bounds)
    if ndim == 1:
        x = np.array([x])
    else:
        x = np.array(x).squeeze()

    lnp = 0
    for i in range(ndim):
        if (x[i] > bounds[i][0]) and (x[i] < bounds[i][1]):
            lnp += 0
        else:
            lnp += -np.inf

    return lnp


def prior_transform_uniform(theta, bounds):

    pt = np.zeros(len(bounds))
    for i, b in enumerate(bounds):
        pt[i] = (b[1] - b[0]) * theta[i] + b[0]

    return pt


def lnprior_normal(x, bounds, data):

    lnp = lnprior_uniform(x, bounds)

    for ii in range(len(x)):
        if data[ii][0] is not None:
            lnp += norm.logpdf(x[ii], data[ii][0], data[ii][1])

    return lnp


def prior_transform_normal(x, bounds, data):

    pt = np.zeros(len(bounds))
    for i, b in enumerate(bounds):
        if data[i][0] is None:
            # uniform prior transform
            pt[i] = (b[1] - b[0]) * x[i] + b[0]
        else:
            # gaussian prior transform
            pt[i] = scipy.stats.norm.ppf(x[i], data[i][0], data[i][1])
    
    return pt


#===========================================================
# Define math functions
#===========================================================

def logsubexp(x1, x2):
    """
    Numerically stable way to compute log(exp(x1) - exp(x2))

    logsubexp(x1, x2) -> log(exp(x1) - exp(x2))

    :param x1: (*float*)
    :param x2: (*float*)

    :returns logsubexp(x1, x2): (*float*)
    """

    if x1 <= x2:
        return -np.inf
    else:
        return x1 + np.log(1.0 - np.exp(x2 - x1))


#===========================================================
# Define utility functions
#===========================================================


def agp_utility(theta, y, gp, bounds):
    """
    AGP (Adaptive Gaussian Process) utility function, the entropy of the
    posterior distribution. This is what you maximize to find the next x under
    the AGP formalism. Note here we use the negative of the utility function so
    minimizing this is the same as maximizing the actual utility function.

    .. math::

        u(x_*) = \\mu(x_*) + \\frac{1}{2} \\ln(2\\pi e \\sigma(x_*)^2) + \\log p

    See Wang & Li (2017) for derivation/explaination.

    :param x: (*array, required*)
        parameters to evaluate
    :param y: (*array, required*)
        y values to condition the gp prediction on.
    :param gp: (*george GP object, required*)

    :param util: (*float*)
        utility of theta under the gp
    """

    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.inf

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    try:
        util = -(mu + 0.5*np.log(2.0*np.pi*np.e*var))
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return util


def bape_utility(theta, y, gp, bounds):
    """
    BAPE (Bayesian Active Posterior Estimation) utility function.  This is what
    you maximize to find the next theta under the BAPE formalism.  Note here we
    use the negative of the utility function so minimizing this is the same as
    maximizing the actual utility function.  Also, we log the BAPE utility
    function as the log is monotonic so the minima are equivalent.

    .. math::

        u(x_*) = e^{2\\mu(x_*) + \\sigma^2(x_*)} \\left(e^{\\sigma^2(x_*)} - 1 \\right)

    See Kandasamy et al. (2015) for derivation/explaination.

    :param x: (*array, required*)
        parameters to evaluate
    :param y: (*array, required*)
        y values to condition the gp prediction on.
    :param gp: (*george GP object, required*)

    :param util: (*float*)
        utility of theta under the gp
    """
    # If guess isn't allowed by prior, we don't care what the value of the
    # utility function is
    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.inf

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    try:
        util = -((2.0 * mu + var) + logsubexp(var, 0.0))
        # util = -(np.exp(2*mu + var) * (np.exp(var) - 1))
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return util


def jones_utility(theta, y, gp, bounds, zeta=0.01):
    """
    Jones utility function - Expected Improvement derived in Jones et al. (1998)

    .. math::
        EI(x_*) = E(max(f(\\theta) - f(\\theta_{best}),0)) 
        
    where f(theta_best) is the best value of the function so far and 
    theta_best is the best design point

    :param x: (*array, required*)
        parameters to evaluate
    :param y: (*array, required*)
        y values to condition the gp prediction on.
    :param gp: (*george GP object, required*)
    :param zeta: (*float, optional*)
        Exploration parameter. Larger zeta leads to more exploration. Defaults
        to 0.01

    :param util: (*float*)
        utility of theta under the gp
    """

    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.inf

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    try:
        std = np.sqrt(var)

        # Find best value
        yBest = np.max(y)

        # Intermediate quantity
        if std > 0:
            z = (mu - yBest - zeta) / std
        else:
            return 0.0

        # Standard normal CDF of z
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)

        util = -((mu - yBest - zeta) * cdf + std * pdf)
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return util


def assign_utility(algorithm):

    # Assign utility function
    if algorithm == "bape":
        utility = bape_utility
    elif algorithm == "agp":
        utility = agp_utility
    elif algorithm == "alternate":
        # If alternate, AGP on even, BAPE on odd
        utility = agp_utility
    elif algorithm == "jones":
        utility = jones_utility
    else:
        errMsg = "Unknown algorithm. Valid options: bape, agp, jones, or alternate."
        raise ValueError(errMsg)

    return utility


def minimize_objective(obj_fn, y, gp, bounds=None, nopt=1, method="nelder-mead",
                       t0=None, ps=None, args=None, options=None, max_iter=100):
    """
    Optimize the active learning objective function
    """
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Initialize options
    if options is None:
        options = {}
    if str(method).lower() == "nelder-mead":
        options["adaptive"] = True
    # options["maxiter"] = 20
    # options["disp"] = True

    # Get prior sampler
    if ps is None:
        ps = partial(prior_sampler, bounds=bounds)

    bound_methods = ["nelder-mead", "l-bfgs", "tnc", "slsqp", "powell", "trust-constr"]
    # scipy >= 1.5 should be installed to use bounds in optimization
    if str(method).lower() not in bound_methods:
        msg = f"Optimization method {method} does not allow bounds."
        msg += "Recommended bound optimization methods: "
        msg += "Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr."
        print(msg)

    # arguments for objective function
    if args is None:
        args = (y, gp, bounds)
        
    if t0 is None:
        t0 = ps(nsample=1).flatten()

    # Containers
    res = []
    objective = []

    # Loop over optimization calls
    for ii in range(nopt):

        # Keep minimizing until a valid solution is found
        test_iter = 0
        while True:

            if test_iter > 0:
                # Try a new initialization point
                t0 = ps(nsample=1).flatten()
                
            # Too many iterations
            if test_iter >= max_iter:
                err_msg = "ERROR: Cannot find a valid solution to objective function minimization.\n" 
                err_msg += "Current iterations: %d\n" % test_iter
                err_msg += "Maximum iterations: %d\n" % max_iter
                err_msg += "Try increasing the number of initial training samples.\n"
                raise RuntimeError(err_msg)
            
            test_iter += 1

            # Minimize the function
            try:
                # warnings.simplefilter("ignore")
                tmp = minimize(obj_fn, np.array(t0).flatten(), args=args, bounds=tuple(bounds), method=method, options=options)
            except Exception as e:
                print("t0:", t0)
                print("bounds:", tuple(bounds))
                print(f"Error optimizing {obj_fn.__name__}: {e}")
                break

            x_opt = tmp.x 
            f_opt = tmp.fun

            # If solution is finite and allowed by the prior, save
            if np.all(np.isfinite(x_opt)) and np.all(np.isfinite(f_opt)):
                if np.isfinite(lnprior_uniform(x_opt, bounds)):
                    res.append(x_opt)
                    objective.append(f_opt)
                    break
                else:
                    print("Warning: Utility function optimization prior fail", x_opt)
            else:
                print("Warning: Utility function optimization infinite fail", x_opt, f_opt)
        
        # end loop

    # Return value that minimizes objective function out of all minimizations
    best_ind = np.argmin(objective)
    theta_best = np.array(res)[best_ind]
    obj_best = objective[best_ind]

    return theta_best, obj_best
