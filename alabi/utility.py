# -*- coding: utf-8 -*-
"""
:py:mod:`utility.py`
----------------------------------------

Utility functions in terms of usefulness, e.g. minimizing GP utility functions
or computing KL divergences, and the GP utility functions, e.g. the bape utility.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from skopt.space import Space
from skopt.sampler import Sobol, Lhs, Halton, Hammersly, Grid
import multiprocessing as mp
import warnings
import time
import tqdm

__all__ = ["agp_utility", 
           "bape_utility", 
           "jones_utility",
           "minimize_objective", 
           "prior_sampler", 
           "eval_fn", 
           "lnprior_uniform",
           "prior_transform_uniform"]


#===========================================================
# Define sampling functions
#===========================================================

def prior_sampler(bounds=None, nsample=1, sampler='uniform'):
    """
    Hypercube sampling function which draws ``nsample`` within given ``bounds``. 
    Wrapper around ``scikit-optimize`` sampling methods. For more info see:
    https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html

    :param bounds: *(array, required)* 
        Array of ``(min,max)`` bounds for each dimension of the prior.

    :param nsample: *(int, optional)* 
        Defaults to ``nsample=1``.

    :param sampler: *(str, optional)* 
        Defaults to ``'uniform'``. Options:
            ``'uniform'``,
            ``'sobol'``,
            ``'lhs'``,
            ``'halton'``,
            ``'hammersly'``,
            ``'grid'``
    """

    ndim = len(bounds)
    space = Space(np.array(bounds, dtype='float'))
    
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

    
def eval_fn(fn, theta, ncore=mp.cpu_count()):

    t0 = time.time()

    if ncore <= 1:
        y = np.zeros(theta.shape[0])
        for ii, tt in tqdm.tqdm(enumerate(theta)):
            y[ii] = fn(tt)
    else:
        with mp.Pool(ncore) as p:
            y = np.array(p.map(fn, theta)).squeeze()

    tf = time.time()
    print(f"Computed {len(theta)} function evaluations: {np.round(tf - t0)}s \n")

    return y


def lnprior_uniform(x, bounds):

    ndim = len(bounds)
    if ndim == 1:
        x = np.array([x])
    else:
        x = x.squeeze()

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

#===========================================================
# Define math functions
#===========================================================

def logsubexp(x1, x2):
    """
    Numerically stable way to compute log(exp(x1) - exp(x2))

    logsubexp(x1, x2) -> log(exp(x1) - exp(x2))

    Parameters
    ----------
    x1 : float
    x2 : float

    Returns
    -------
    logsubexp(x1, x2)
    """

    if x1 <= x2:
        return -np.inf
    else:
        return x1 + np.log(1.0 - np.exp(x2 - x1))


#===========================================================
# Define utility functions
#===========================================================


def agp_utility(theta, y, gp):
    """
    AGP (Adaptive Gaussian Process) utility function, the entropy of the
    posterior distribution. This is what you maximize to find the next x under
    the AGP formalism. Note here we use the negative of the utility function so
    minimizing this is the same as maximizing the actual utility function.

    See Wang & Li (2017) for derivation/explaination.

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object

    Returns
    -------
    util : float
        utility of theta under the gp
    """

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

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object

    Returns
    -------
    util : float
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

    if var <= 0:
        return np.inf

    try:
        util = -((2.0 * mu + var) + logsubexp(var, 0.0))
        # util = -(np.exp(2*mu + var) * (np.exp(var) - 1))
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    # print(mu, var, logsubexp(var, 0.0), util)
    # print('theta:', theta, 'util', util)

    return util


def jones_utility(theta, y, gp, zeta=0.01):
    """
    Jones utility function - Expected Improvement derived in Jones et al. (1998)
    EI(x) = E(max(f(theta) - f(thetaBest),0)) where f(thetaBest) is the best
    value of the function so far and thetaBest is the best design point

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object
    zeta : float, optional
        Exploration parameter. Larger zeta leads to more exploration. Defaults
        to 0.01

    Returns
    -------
    util : float
        utility of theta under the gp
    """

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


def minimize_objective(obj_fn, y, gp, bounds=None, nopt=1, method="nelder-mead",
                       t0=None, args=None, options={}, max_iter=100):

    # Initialize options
    if str(method).lower() == "nelder-mead" and options is None:
        options["adaptive"] = True
    # options["maxiter"] = 20
    # options["disp"] = True

    bound_methods = ["nelder-mead", "l-bfgs", "tnc", "slsqp", "powell", "trust-constr"]
    # scipy >= 1.5 should be installed to use bounds in optimization
    if str(method).lower() not in bound_methods:
        msg = f"Optimization method {method} does not allow bounds."
        msg += "Recommended bound optimization methods: "
        msg += "Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr."
        print(msg)

    # arguments for objective function
    if args is None:
        args = ()

    # Containers
    res = []
    objective = []

    # Loop over optimization calls
    for ii in range(nopt):

        # Keep minimizing until a valid solution is found
        test_iter = 0
        while True:

            if t0 is None:
                t0 = prior_sampler(nsample=1, bounds=bounds)

            # Too many iterations
            if test_iter >= max_iter:
                err_msg = "ERROR: Cannot find a valid solution to objective function minimization.\n" 
                err_msg += "Current iterations: %d\n" % test_iter
                err_msg += "Maximum iterations: %d\n" % max_iter
                err_msg += "Try increasing the number of initial training samples.\n"
                raise RuntimeError(err_msg)

            # Minimize the function
            try:
                warnings.simplefilter("ignore")
                tmp = minimize(obj_fn, t0, args=args, bounds=tuple(bounds),
                               method=method, options=options)
            except:
                raise Warning('Objective function optimization failed')

            x_opt = tmp.x 
            f_opt = tmp.fun

            # If solution is finite and allowed by the prior, save
            if np.all(np.isfinite(x_opt)) and np.all(np.isfinite(f_opt)):
                if np.isfinite(lnprior_uniform(x_opt, bounds)):
                    res.append(x_opt)
                    objective.append(f_opt)

                    break
                else:
                    print('Utility function optimization prior fail', x_opt)
            else:
                print('Utility function optimization infinite fail', x_opt, f_opt)

            # Try a new initialization point
            t0 = prior_sampler(nsample=1, bounds=bounds)
            
            test_iter += 1
        # end loop

    # Return value that minimizes objective function out of all minimizations
    best_ind = np.argmin(objective)
    theta_best = np.array(res)[best_ind]
    obj_best = objective[best_ind]

    return theta_best, obj_best
