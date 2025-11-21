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
from scipy.special import betainc, betaincinv
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

# Import parallel utilities for MPI-safe multiprocessing
try:
    from alabi import parallel_utils
except ImportError:
    # Fallback if parallel_utils is not available
    parallel_utils = None
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
           "BetaWarpingFunction",
           "beta_warping_transformer"
           ]


#===========================================================
# Define sampling functions
#===========================================================

def prior_sampler(bounds=None, nsample=1, sampler='uniform', random_state=None):
    """
    Sample from parameter space using various quasi-random sampling methods.
    
    This function generates samples within specified bounds using different sampling
    strategies from the scikit-optimize library. It provides a unified interface to
    various space-filling designs commonly used in Bayesian optimization and 
    surrogate modeling.
    
    :param bounds: (*array-like of shape (ndim, 2)*)
        Array of (min, max) bounds for each parameter dimension. Each row specifies
        the lower and upper bounds for one parameter.
        
    :param nsample: (*int, optional*)
        Number of samples to generate. Default is 1.
        
    :param sampler: (*{'uniform', 'sobol', 'lhs', 'halton', 'hammersly', 'grid'}, optional*)
        Sampling method to use:
        
        - 'uniform': Pseudo-random uniform sampling
        - 'sobol': Sobol sequence (quasi-random, good space-filling)
        - 'lhs': Latin Hypercube Sampling (stratified sampling)
        - 'halton': Halton sequence (quasi-random, low discrepancy)
        - 'hammersly': Hammersley sequence (quasi-random)
        - 'grid': Regular grid sampling
        
        Default is 'uniform'.
        
    :param random_state: (*int, RandomState instance or None, optional*)
        Random state for reproducible sampling. If None, uses a different random
        seed each time to avoid clustering. Default is None.
    
    :returns: *ndarray of shape (nsample, ndim)*
        Array of parameter samples within the specified bounds.
    
    :raises ValueError:
        If an invalid sampler method is specified.
    
    **Notes**
    
    Quasi-random samplers (sobol, halton, hammersley) provide better space-filling
    properties than pseudo-random uniform sampling, which is beneficial for:
    - Initial design of experiments
    - Training surrogate models
    - Global optimization
    
    Latin Hypercube Sampling ensures each parameter dimension is stratified,
    providing good marginal coverage even with small sample sizes.
    
    For optimization starting points, consider using 'sobol' or 'lhs' instead of
    'uniform' to avoid clustering issues when generating single samples repeatedly.
    
    **Examples**
    
    Generate uniform random samples:
    
    .. code-block:: python
    
        bounds = [(-1, 1), (0, 10)]
        samples = prior_sampler(bounds, nsample=5, sampler='uniform')
        print(samples.shape)  # (5, 2)
    
    Use Sobol sequence for better space-filling:
    
    .. code-block:: python
    
        samples = prior_sampler(bounds, nsample=100, sampler='sobol')
    
    Latin Hypercube Sampling for stratified design:
    
    .. code-block:: python
    
        samples = prior_sampler(bounds, nsample=20, sampler='lhs')
    
    **References**
    
    For more information on sampling methods, see:
    https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html
    """

    ndim = len(bounds)
        
    # space_bounds = [Categorical(categories=(np.float64(bounds[ii][0]), np.float64(bounds[ii][1]))) for ii in range(ndim)]
    space_bounds = [Real(bounds[ii][0], bounds[ii][1], dtype='float') for ii in range(ndim)]
    space = Space(space_bounds)
    
    if sampler == 'uniform':
        # For uniform sampling, ensure proper random state management to avoid clustering
        if random_state is None:
            # Use a time-based seed to avoid clustering when called repeatedly
            import time
            random_state = int(time.time() * 1000000) % (2**32)
        samples = space.rvs(nsample, random_state=random_state)

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
    """
    Evaluate a function at multiple parameter points with optional parallelization.
    
    This utility function provides a convenient interface for evaluating expensive
    functions (like likelihood functions or forward models) at multiple parameter
    values, with automatic parallelization when multiple cores are available.
    
    :param fn: (*callable*)
        Function to evaluate at each parameter point. Should have signature
        fn(theta_i) where theta_i is array of shape (ndim,) and return a scalar.
        
    :param theta: (*array-like of shape (npoints, ndim)*)
        Array of parameter points at which to evaluate the function. Each row
        represents one parameter vector.
        
    :param ncore: (*int, optional*)
        Number of CPU cores to use for parallel evaluation. If ncore <= 1,
        uses serial evaluation with progress bar. If ncore > 1, uses
        multiprocessing.Pool for parallel evaluation. Default is all available cores.
        
    :returns: *ndarray of shape (npoints,)*
        Function values evaluated at each parameter point. Order matches the
        input theta array.
        
    **Notes**
    
    This function is particularly useful for:
    - Initial sampling of expensive functions for surrogate model training
    - Batch evaluation of likelihood functions
    - Testing surrogate model accuracy on validation sets
    
    The function automatically prints timing information and handles both serial
    and parallel execution modes. For serial execution, a progress bar is shown
    using tqdm.
    
    **Examples**
    
    Evaluate a simple function at multiple points:
    
    .. code-block:: python
    
        def quadratic(x):
            return np.sum(x**2)
        theta = np.random.rand(100, 2)
        y = eval_fn(quadratic, theta, ncore=1)
    
    Parallel evaluation of expensive function:
    
    .. code-block:: python
    
        def expensive_fn(x):
            # Simulate expensive computation
            time.sleep(0.1)
            return np.sum(x**2)
        y = eval_fn(expensive_fn, theta, ncore=4)
    """

    t0 = time.time()

    if ncore <= 1:
        y = np.zeros(theta.shape[0])
        for ii, tt in tqdm.tqdm(enumerate(theta)):
            y[ii] = fn(tt)
    else:
        # Use MPI-safe multiprocessing
        if parallel_utils is not None:
            y = parallel_utils.safe_pool_map(fn, theta, ncore)
        else:
            # Fallback to original implementation
            with mp.Pool(ncore) as p:
                y = np.array(p.map(fn, theta))
    y = np.array(y)

    tf = time.time()
    print(f"Computed {len(theta)} function evaluations: {np.round(tf - t0)}s \n")

    try:
        return y.squeeze()
    except:
        return y


def lnprior_uniform(x, bounds):
    """
    Evaluate log-probability density of uniform prior distribution.
    
    This function computes the log-probability density for a uniform (flat)
    prior distribution within specified bounds. Points outside the bounds
    receive log-probability of negative infinity.
    
    :param x: (*array-like of shape (ndim,) or float*)
        Parameter values at which to evaluate the log-prior. For scalar input,
        assumes 1D parameter space.
        
    :param bounds: (*array-like of shape (ndim, 2)*)
        Parameter bounds as [(min, max), ...] for each dimension.
        
    :returns: *float*
        Log-probability density. Returns 0.0 if all parameters are within bounds,
        -np.inf if any parameter is outside bounds.
        
    **Notes**
    
    For a uniform distribution on [a, b], the probability density is 1/(b-a)
    and the log-probability density is -log(b-a). However, this function 
    returns 0.0 for in-bounds points since constant offsets don't affect
    relative probabilities in MCMC sampling.
    
    **Examples**
    
    Check if point is within bounds:
    
    .. code-block:: python
    
        bounds = [(-1, 1), (0, 2)]
        x = [0.5, 1.0]
        lnp = lnprior_uniform(x, bounds)  # Returns 0.0
    
    Point outside bounds:
    
    .. code-block:: python
    
        x = [2.0, 1.0]  # First parameter out of bounds
        lnp = lnprior_uniform(x, bounds)  # Returns -inf
    """

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
    """
    Transform uniform random variables to parameter space with specified bounds.
    
    This function implements the inverse CDF transformation for uniform distributions,
    mapping from the unit hypercube [0,1]^ndim to the parameter space with given bounds.
    It is commonly used in nested sampling algorithms like dynesty and UltraNest.
    
    :param theta: (*array-like*)
        Random variables uniformly distributed on [0,1]. Can be:
        - 1D array of shape (ndim,) for a single parameter vector
        - 2D array of shape (nsamples, ndim) for multiple parameter vectors
        
    :param bounds: (*array-like of shape (ndim, 2)*)
        Parameter bounds as [(min, max), ...] for each dimension.
        
    :returns: *ndarray*
        Transformed parameter values within the specified bounds.
        - If input is 1D: returns 1D array of shape (ndim,)
        - If input is 2D: returns 2D array of shape (nsamples, ndim)
        
    **Notes**
    
    The transformation for each dimension i is:
    
    .. math::
        \\theta'_i = (b_{i,\\text{max}} - b_{i,\\text{min}}) \\theta_i + b_{i,\\text{min}}
    
    where b_{i,min} and b_{i,max} are the bounds for dimension i.
    
    This is the inverse of the uniform CDF, mapping uniform random variables
    on [0,1] to uniform random variables on [a,b].
    
    **Examples**
    
    Transform single parameter vector:
    
    .. code-block:: python
    
        bounds = [(-2, 2), (0, 10)]
        theta_unit = [0.25, 0.8]  # Single vector
        theta_params = prior_transform_uniform(theta_unit, bounds)
        print(theta_params)  # [-1.0, 8.0]
    
    Transform multiple parameter vectors:
    
    .. code-block:: python
    
        theta_unit = [[0.25, 0.8], [0.5, 0.2]]  # Multiple vectors
        theta_params = prior_transform_uniform(theta_unit, bounds)
        print(theta_params)  # [[-1.0, 8.0], [0.0, 2.0]]
    
    Use with nested sampling:
    
    .. code-block:: python
    
        def my_prior_transform(u):
            return prior_transform_uniform(u, bounds)
        # Pass to dynesty/UltraNest sampler
    """
    
    # Convert to numpy array and handle input validation
    theta = np.asarray(theta, dtype=float)
    bounds = np.asarray(bounds)
    
    # Determine if input is 1D or 2D
    if theta.ndim == 1:
        # Single parameter vector
        ndim = len(theta)
        if len(bounds) != ndim:
            raise ValueError(f"Bounds length ({len(bounds)}) must match theta dimensions ({ndim})")
        
        pt = np.zeros(ndim)
        for i, b in enumerate(bounds):
            pt[i] = (b[1] - b[0]) * theta[i] + b[0]
        return pt
        
    elif theta.ndim == 2:
        # Multiple parameter vectors
        nsamples, ndim = theta.shape
        if len(bounds) != ndim:
            raise ValueError(f"Bounds length ({len(bounds)}) must match theta dimensions ({ndim})")
        
        pt = np.zeros((nsamples, ndim))
        for i, b in enumerate(bounds):
            pt[:, i] = (b[1] - b[0]) * theta[:, i] + b[0]
        return pt
        
    else:
        raise ValueError(f"theta must be 1D or 2D array, got {theta.ndim}D array with shape {theta.shape}")


def lnprior_normal(x, bounds, data):

    lnp = lnprior_uniform(x, bounds)

    for ii in range(len(x)):
        if data[ii][0] is not None:
            lnp += norm.logpdf(x[ii], data[ii][0], data[ii][1])

    return lnp


def prior_transform_normal(x, bounds, data):
    """
    Transform uniform random variables to parameter space with mixed prior distributions.
    
    This function implements prior transformations supporting both uniform and Gaussian
    priors for different parameters. It maps from the unit hypercube [0,1]^ndim to
    the parameter space, handling each dimension according to its specified prior type.
    
    :param x: (*array-like*)
        Random variables uniformly distributed on [0,1]. Can be:
        - 1D array of shape (ndim,) for a single parameter vector
        - 2D array of shape (nsamples, ndim) for multiple parameter vectors
        
    :param bounds: (*array-like of shape (ndim, 2)*)
        Parameter bounds as [(min, max), ...] for each dimension. Used for uniform priors.
        
    :param data: (*list of tuples*)
        Prior specification for each dimension as [(mean, std), ...]. 
        - If data[i] = (None, None): use uniform prior with bounds[i]
        - If data[i] = (mean, std): use Gaussian prior with specified mean and standard deviation
        
    :returns: *ndarray*
        Transformed parameter values according to their prior distributions.
        - If input is 1D: returns 1D array of shape (ndim,)
        - If input is 2D: returns 2D array of shape (nsamples, ndim)
        
    **Notes**
    
    For uniform priors (when data[i][0] is None):
    
    .. math::
        \\theta'_i = (b_{i,\\text{max}} - b_{i,\\text{min}}) x_i + b_{i,\\text{min}}
    
    For Gaussian priors (when data[i] = (μ, σ)):
    
    .. math::
        \\theta'_i = \\Phi^{-1}(x_i; \\mu_i, \\sigma_i)
    
    where Φ^{-1} is the inverse normal CDF (percent-point function).
    
    **Examples**
    
    Mixed uniform and Gaussian priors:
    
    .. code-block:: python
    
        bounds = [(-2, 2), (0, 10)]
        data = [(None, None), (5.0, 1.0)]  # uniform, then Gaussian(5, 1)
        x_unit = [0.5, 0.84]  # Single vector
        x_params = prior_transform_normal(x_unit, bounds, data)
        print(x_params)  # [0.0, ~6.0] (second value from normal inverse CDF)
    
    Vectorized transformation:
    
    .. code-block:: python
    
        x_unit = [[0.5, 0.84], [0.25, 0.16]]  # Multiple vectors
        x_params = prior_transform_normal(x_unit, bounds, data)
        # Returns 2D array with transformed parameters
    """
    
    # Convert to numpy array and handle input validation
    x = np.asarray(x, dtype=float)
    bounds = np.asarray(bounds)
    
    # Determine if input is 1D or 2D
    if x.ndim == 1:
        # Single parameter vector
        ndim = len(x)
        if len(bounds) != ndim or len(data) != ndim:
            raise ValueError(f"Bounds length ({len(bounds)}) and data length ({len(data)}) "
                           f"must match x dimensions ({ndim})")
        
        pt = np.zeros(ndim)
        for i, b in enumerate(bounds):
            if data[i][0] is None:
                # uniform prior transform
                pt[i] = (b[1] - b[0]) * x[i] + b[0]
            else:
                # gaussian prior transform
                pt[i] = scipy.stats.norm.ppf(x[i], data[i][0], data[i][1])
        return pt
        
    elif x.ndim == 2:
        # Multiple parameter vectors
        nsamples, ndim = x.shape
        if len(bounds) != ndim or len(data) != ndim:
            raise ValueError(f"Bounds length ({len(bounds)}) and data length ({len(data)}) "
                           f"must match x dimensions ({ndim})")
        
        pt = np.zeros((nsamples, ndim))
        for i, b in enumerate(bounds):
            if data[i][0] is None:
                # uniform prior transform
                pt[:, i] = (b[1] - b[0]) * x[:, i] + b[0]
            else:
                # gaussian prior transform
                pt[:, i] = scipy.stats.norm.ppf(x[:, i], data[i][0], data[i][1])
        return pt
        
    else:
        raise ValueError(f"x must be 1D or 2D array, got {x.ndim}D array with shape {x.shape}")


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


def agp_utility(theta, predict_gp, bounds):
    """
    Compute the AGP (Adaptive Gaussian Process) utility function based on posterior entropy.
    
    AGP is an information-theoretic acquisition function that measures the entropy
    of the Gaussian Process posterior distribution. It balances the GP mean prediction
    with the uncertainty (variance), preferring regions with high predicted values
    and high uncertainty.
    
    :param theta: (*array-like of shape (ndim,)*)
        Parameter values at which to evaluate the utility function.
        
    :param y: (*array-like of shape (nsamples,)*)
        Observed function values at training points, used to condition the GP.
        
    :param gp: (*george.GP*)
        Trained Gaussian Process model. The GP will be computed if not already done.
        
    :param bounds: (*array-like of shape (ndim, 2)*)
        Parameter bounds as [(min, max), ...] for each dimension. Used to check
        if theta is within the prior support.
    
    :returns: *float*
        Negative AGP utility value. The negative is used so that minimizing
        this function is equivalent to maximizing the actual utility.
    
    **Notes**
    
    The AGP utility function is defined as:
    
    .. math::
        u(\\theta) = \\mu(\\theta) + \\frac{1}{2} \\ln(2\\pi e \\sigma^2(\\theta))
    
    This represents the entropy of the posterior predictive distribution at θ.
    The utility encourages sampling where:
    - The mean prediction μ(θ) is high (exploitation)
    - The predictive variance σ²(θ) is high (exploration)
    
    AGP provides a different balance compared to other acquisition functions:
    - More exploitative than BAPE (focuses on high mean regions)
    - Less optimization-focused than Expected Improvement
    - Naturally balances exploration and exploitation through entropy
    
    :raises ValueError:
        If the utility computation results in invalid values (e.g., negative variance).
    
    **Examples**
    
    Evaluate AGP utility at a test point:
    
    .. code-block:: python
    
        theta_test = np.array([0.5, 1.2])
        utility = agp_utility(theta_test, y_train, gp, bounds)
    
    Find next point by minimizing negative utility:
    
    .. code-block:: python
    
        from scipy.optimize import minimize
        result = minimize(lambda x: agp_utility(x, y_train, gp, bounds), x0)
        next_point = result.x
    
    **References**
    
    Wang & Li (2017): "Adaptive Gaussian Process Approximation for Bayesian 
    Inference with Expensive Likelihood Functions", Neural Computation, 30, 3072-3094.
    """

    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.inf

    # if not gp.computed:
    #     gp.compute(theta)

    mu, var = predict_gp(theta.reshape(1,-1))

    try:
        util = -(mu + 0.5*np.log(2.0*np.pi*np.e*var))
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return float(util)


def bape_utility(theta, predict_gp, bounds):
    """
    Compute the BAPE (Bayesian Active Posterior Estimation) utility function.
    
    BAPE is an active learning acquisition function designed for posterior exploration
    rather than optimization. It identifies regions where the GP variance is high
    relative to the mean, promoting exploration of the parameter space. The utility
    is computed in log-form for numerical stability.
    
    :param theta: Parameter values at which to evaluate the utility function.
    :type theta: *array-like of shape (ndim,)*
    :param y: Observed function values at training points, used to condition the GP.
    :type y: *array-like of shape (nsamples,)*
    :param gp: Trained Gaussian Process model. Must have been computed (gp.computed=True).
    :type gp: *george.GP*
    :param bounds: Parameter bounds as [(min, max), ...] for each dimension. Used to check
        if theta is within the prior support.
    :type bounds: *array-like of shape (ndim, 2)*
    
    :returns: Negative log-BAPE utility value. The negative is used so that minimizing
        this function is equivalent to maximizing the actual utility.
    :rtype: *float*
    
    .. note::
        
        The BAPE utility function is defined as:
        
        .. math::
            
            u(\\theta) = e^{2\\mu(\\theta) + \\sigma^2(\\theta)} \\left(e^{\\sigma^2(\\theta)} - 1 \\right)
        
        This function returns the negative logarithm of the utility for numerical stability:
        
        .. math::
            
            -\\log u(\\theta) = -\\left(2\\mu(\\theta) + \\sigma^2(\\theta) + \\log(e^{\\sigma^2(\\theta)} - 1)\\right)
        
        BAPE is particularly effective for:
        
        - Exploring multi-modal posteriors
        - Reducing uncertainty in posterior estimates
        - Active learning when the goal is posterior characterization
        
        Unlike optimization-focused acquisitions (e.g., Expected Improvement), BAPE
        prioritizes exploration over exploitation, making it less suitable for finding
        global optima but excellent for posterior mapping.
    
    :raises ValueError: If the utility computation results in invalid values (e.g., negative variance).
    :raises RuntimeError: If the GP has not been computed before calling this function.
    
    **Examples**
    
    Evaluate BAPE utility at a test point:
    
    .. code-block:: python
        
        >>> theta_test = np.array([0.5, 1.2])
        >>> utility = bape_utility(theta_test, y_train, gp, bounds)
    
    Find next point by minimizing negative utility:
    
    .. code-block:: python
        
        >>> from scipy.optimize import minimize
        >>> result = minimize(lambda x: bape_utility(x, y_train, gp, bounds), x0)
        >>> next_point = result.x
    
    **References**
    
    Kandasamy et al. (2015): "Query efficient posterior estimation in scientific 
    experiments via Bayesian active learning", Artificial Intelligence, 243, 45-56.
    """
    
    # Ensure theta is properly shaped
    theta = np.asarray(theta).flatten()
    
    # Check that theta is in bounds
    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.inf

    mu, var = predict_gp(theta.reshape(1,-1))

    try:
        util = -((2.0 * mu + var) + logsubexp(var, 0.0))
        # util = -(np.exp(2*mu + var) * (np.exp(var) - 1))
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return float(util)


def jones_utility(theta, predict_gp, bounds, y_best, zeta=0.01):
    """
    Compute the Expected Improvement (EI) acquisition function.
    
    This function implements the Expected Improvement criterion from Jones et al. (1998),
    which balances exploitation (sampling where the mean is high) with exploration
    (sampling where the uncertainty is high). Unlike BAPE, this acquisition function
    is designed specifically for global optimization.
    
    :param theta: Parameter values at which to evaluate the utility function.
    :type theta: *array-like of shape (ndim,)*
    :param y: Observed function values at training points. The maximum value is used
        as the current best (f_best).
    :type y: *array-like of shape (nsamples,)*
    :param gp: Trained Gaussian Process model. Must have been computed (gp.computed=True).
    :type gp: *george.GP*
    :param bounds: Parameter bounds as [(min, max), ...] for each dimension. Used to check
        if theta is within the prior support.
    :type bounds: *array-like of shape (ndim, 2)*
    :param zeta: Exploration parameter controlling the trade-off between exploitation
        and exploration. Larger values promote more exploration:
        
        - zeta = 0: Pure exploitation (greedy)
        - zeta > 0: Balanced exploration/exploitation
        - zeta >> 0: Pure exploration
        
        Default is 0.01.
    :type zeta: *float, optional*
    
    :returns: Negative Expected Improvement value. The negative is used so that minimizing
        this function is equivalent to maximizing the Expected Improvement.
    :rtype: *float*
    
    .. note::
        
        The Expected Improvement is defined as:
        
        .. math::
            
            EI(\\theta) = \\mathbb{E}[\\max(f(\\theta) - f_{\\text{best}} - \\zeta, 0)]
        
        where f_best is the current best observed value. For a Gaussian predictive
        distribution with mean μ and variance σ², this becomes:
        
        .. math::
            
            EI(\\theta) = (\\mu - f_{\\text{best}} - \\zeta) \\Phi(z) + \\sigma \\phi(z)
        
        where z = (μ - f_best - ζ)/σ, Φ is the standard normal CDF, and φ is the
        standard normal PDF.
        
        Expected Improvement is particularly effective for:
        
        - Global optimization problems
        - Finding the maximum of expensive functions
        - Balancing local and global search
        
        Compared to BAPE, Expected Improvement focuses on exploitation (finding optima)
        rather than exploration (mapping the entire posterior).
    
    :raises ValueError: If the utility computation results in invalid values.
    :raises RuntimeError: If the GP has not been computed before calling this function.
    
    **Examples**
    
    Evaluate Expected Improvement at a test point:
    
    .. code-block:: python
        
        >>> theta_test = np.array([0.5, 1.2])
        >>> ei_value = jones_utility(theta_test, y_train, gp, bounds)
    
    Find next point for optimization:
    
    .. code-block:: python
        
        >>> from scipy.optimize import minimize
        >>> result = minimize(lambda x: jones_utility(x, y_train, gp, bounds, zeta=0.01), x0)
        >>> next_point = result.x
    
    Use higher exploration parameter:
    
    .. code-block:: python
        
        >>> ei_explore = jones_utility(theta_test, y_train, gp, bounds, zeta=0.1)
    
    **References**
    
    Jones et al. (1998): "Efficient global optimization of expensive black-box 
    functions", Journal of Global Optimization, 13, 455-492.
    """

    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.inf

    # Only works if the GP object has been computed, otherwise you messed up
    mu, var = predict_gp(theta.reshape(1,-1))

    try:
        std = np.sqrt(var)

        # Intermediate quantity
        if std > 0:
            z = (mu - y_best - zeta) / std
        else:
            return 0.0

        # Standard normal CDF of z
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)

        util = -((mu - y_best - zeta) * cdf + std * pdf)
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return float(util)


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
        errMsg = "Unknown algorithm. Valid options: bape, agp, jones, kg (knowledge_gradient), or alternate."
        raise ValueError(errMsg)

    return utility


def minimize_objective_single(idx, obj_fn, bounds, starting_point, method, options, grad_obj_fn=None):
    """
    Single optimization run - used for parallelization.
    
    This is a helper function that performs one optimization attempt with a specific
    starting point. It is called serially by the main minimize_objective function
    to handle multiple optimization restarts.
    
    :param idx: Index of the optimization run.
    :type idx: *int*
    :param obj_fn: Objective function to minimize.
    :type obj_fn: *callable*
    :param bounds: Parameter bounds.
    :type bounds: *list*
    :param set_bounds: Bounds for optimization method.
    :type set_bounds: *list or None*
    :param starting_points: Pre-generated starting points for optimization.
    :type starting_points: *ndarray*
    :param method: Optimization method.
    :type method: *str*
    :param options: Optimization options.
    :type options: *dict*
    :param n_attempts: Maximum number of attempts.
    :type n_attempts: *int*
    :param grad_obj_fn: Gradient function (optional).
    :type grad_obj_fn: *callable or None*
    
    :returns: Optimization result object.
    :rtype: *scipy.optimize.OptimizeResult*
    """
    
    # Minimize the function
    tmp = minimize(fun=obj_fn, 
                    x0=np.array(starting_point).flatten(), 
                    jac=grad_obj_fn, 
                    bounds=bounds, 
                    method=method, 
                    options=options)

    x_opt = tmp.x 
    f_opt = tmp.fun
    
    # If solution is finite and allowed by the prior, save
    if np.all(np.isfinite(x_opt)) and np.all(np.isfinite(f_opt)):
        if np.isfinite(lnprior_uniform(x_opt, bounds)):
            return x_opt, f_opt
        else:
            print("Warning: Utility function optimization prior fail", x_opt)
            return np.nan, np.nan
    else:
        print("Warning: Utility function optimization infinite fail", x_opt, f_opt)
        return np.nan, np.nan


def minimize_objective(obj_fn, bounds=None, nopt=1, method="l-bfgs-b",
                       ps=None, options=None, grad_obj_fn=None):
    """
    Find the global minimum of an acquisition function using multiple restarts.
    
    This function optimizes acquisition functions (BAPE, Jones/EI, etc.) to select
    the next point for active learning. Multiple optimization restarts with different
    initial points help avoid local minima, which is crucial for acquisition function
    optimization where the landscape can be highly multi-modal.
    
    :param obj_fn: Objective function to minimize. Should have signature obj_fn(theta) and
        return a scalar value. Typically an acquisition function like BAPE or EI.
    :type obj_fn: *callable*
    :param grad_obj_fn: Gradient of the objective function. Should have signature grad_obj_fn(theta)
        and return array of shape (ndim,). If None, uses finite differences.
    :type grad_obj_fn: *callable or None*
    :param bounds: Parameter bounds as [(min, max), ...] for each dimension. Used both
        for optimization constraints and generating initial points.
    :type bounds: *array-like of shape (ndim, 2)*
    :param nopt: Number of optimization restarts with different initial points. More
        restarts increase the chance of finding the global minimum but increase
        computational cost. Default is 1.
    :type nopt: *int, optional*
    :param method: Scipy optimization method to use. Common choices:
        
        - "l-bfgs-b": Quasi-Newton with bounds (default, works well with gradients)
        - "nelder-mead": Simplex method (gradient-free, robust)
        - "tnc": Truncated Newton with bounds
        - "slsqp": Sequential Least Squares Programming
        
        Default is "l-bfgs-b".
    :type method: *str, optional*
    :param ps: Prior sampling function with signature ps(nsample=1). Used to generate
        initial points for optimization restarts. If None, uses uniform sampling
        within bounds. Default is None.
    :type ps: *callable or None, optional*
    :param options: Additional options passed to scipy.optimize.minimize. Common options:
        
        - "max_iter": Maximum iterations (default: 50)
        - "ftol": Function tolerance for convergence
        - "gtol": Gradient tolerance for convergence
        
        Default is {"max_iter": 50}.
    :type options: *dict or None, optional*
    :param grad_obj_fn: Gradient of the objective function. If provided, can significantly
        speed up optimization for methods like l-bfgs-b. Default is None (use finite differences).
    :type grad_obj_fn: *callable or None, optional*
    
    :returns: 
        - **theta_best** (*ndarray of shape (ndim,)*) -- Parameter values that minimize the objective function.
        - **obj_best** (*float*) -- Minimum objective function value achieved.
    :rtype: *tuple*
    
    :raises RuntimeError: If no valid solutions are found after all optimization attempts.
    """
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Initialize options with speed optimizations
    if options is None:
        if method.lower() == "l-bfgs-b":
            # More aggressive settings for L-BFGS-B
            options = {"maxiter": 100, "ftol": 1e-6, "gtol": 1e-5}
        elif method.lower() == "nelder-mead":
            # Faster convergence for Nelder-Mead
            options = {"maxiter": 200, "xatol": 1e-6, "fatol": 1e-6}
        else:
            options = {"maxiter": 50}
    else:
        # Make a copy to avoid modifying the original
        options = options.copy()
        
        # Normalize option names for different optimization methods
        # Handle common user-friendly names vs scipy's internal parameter names
        if "max_iter" in options:
            options["maxiter"] = options.pop("max_iter")
        if "max_eval" in options:
            options["maxfev"] = options.pop("max_eval")
        if "max_fun" in options:
            options["maxfun"] = options.pop("max_fun")
    
    # Add method-specific optimizations for speed
    if str(method).lower() == "nelder-mead":
        options["adaptive"] = True
    elif str(method).lower() == "l-bfgs-b":
        # Ensure we use limited memory for large problems
        if "maxcor" not in options:
            options["maxcor"] = 10

    # Generate all starting points at once to ensure good space-filling distribution
    # This prevents clustering that can occur when generating points individually
    if ps is None:
        # Use Sobol sequence for better space-filling and faster generation
        starting_points = prior_sampler(bounds, nsample=nopt, sampler='sobol')
    else:
        # Use provided sampler to generate all points at once
        starting_points = ps(nsample=nopt)
    
    # Pre-flatten starting points to avoid repeated operations
    starting_points = np.array([pt.flatten() for pt in starting_points])

    # Serial execution 
    min_theta = []
    min_obj = []
    
    for ii in range(nopt):
        minx, miny = minimize_objective_single(ii, obj_fn, bounds, starting_points[ii], method, options, grad_obj_fn)
        min_theta.append(minx)
        min_obj.append(miny)
  
    # Return value that minimizes objective function out of all minimizations
    best_ind = np.argmin(min_obj)
    theta_best = min_theta[best_ind]  
    obj_best = min_obj[best_ind]

    return theta_best, obj_best


class BetaWarpingFunction:
    """
    Swersky et al. 2017 Beta CDF warping function for sklearn FunctionTransformer.
    """
    def __init__(self, alpha=2.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta
        self.minmax_scaler = MinMaxScaler()
        self._is_fitted = False
    
    def transform(self, X):
        if not self._is_fitted:
            self.minmax_scaler.fit(X)
            self._is_fitted = True
        
        X_scaled = self.minmax_scaler.transform(X)
        
        # Apply Beta CDF warping per feature
        X_warped = np.zeros_like(X_scaled)
        for i in range(X_scaled.shape[1]):
            X_warped[:, i] = betainc(self.alpha, self.beta, X_scaled[:, i])
        
        return X_warped
    
    def inverse_transform(self, X):
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        # Apply inverse Beta CDF per feature
        X_unwarped = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_unwarped[:, i] = betaincinv(self.alpha, self.beta, X[:, i])
        
        return self.minmax_scaler.inverse_transform(X_unwarped)
    
    
def beta_warping_transformer(alpha=2.0, beta=2.0):
    """
    Create a scikit-learn FunctionTransformer for Beta CDF warping.
    
    Parameters
    ----------
    alpha : float or array-like
        Shape parameter α for Beta distribution
    beta : float or array-like
        Shape parameter β for Beta distribution
    
    Returns
    -------
    transformer : FunctionTransformer
        Scikit-learn transformer object
    """
    warping_funcs = BetaWarpingFunction(alpha=alpha, beta=beta)
    
    return FunctionTransformer(
        func=warping_funcs.transform,
        inverse_func=warping_funcs.inverse_transform,
        check_inverse=True
    )