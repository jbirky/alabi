# -*- coding: utf-8 -*-
"""
:py:mod:`utility.py`
----------------------------------------

Utility functions in terms of usefulness, e.g. minimizing GP utility functions
or computing KL divergences, and the GP utility functions, e.g. the bape utility.
"""

from curses import nl
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm
from skopt.space import Space
from skopt.space.space import Real
from skopt.sampler import Sobol, Lhs, Halton, Hammersly, Grid
import warnings
from scipy.special import betainc, betaincinv
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

__all__ = ["agp_utility", 
           "bape_utility", 
           "jones_utility",
           "assign_utility",
           "minimize_objective", 
           "prior_sampler", 
           "prior_sampler_normal",
           "lnprior_uniform",
           "prior_transform_uniform",
           "lnprior_normal",
           "prior_transform_normal",
           "BetaWarpingFunction",
           "beta_warping_transformer",
           "NewFunctionTransformer",
           "nlog_scaler", "log_scaler", "no_scaler"
           ]


#===========================================================
# Define data transformation functions
#===========================================================

class NewFunctionTransformer(preprocessing.FunctionTransformer):
    def __init__(self, name, func=None, inverse_func=None, *, validate=False,
                 accept_sparse=False, check_inverse=True, feature_names_out=None,
                 kw_args=None, inv_kw_args=None):
        super().__init__(
            func=func, inverse_func=inverse_func, validate=validate,
            accept_sparse=accept_sparse, check_inverse=check_inverse,
            feature_names_out=feature_names_out, kw_args=kw_args,
            inv_kw_args=inv_kw_args,
        )
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

# Define scaling functions 
def nlog(x): return np.log10(-x)
def nlog_inv(x): return -10**x
def log_scale(x): return np.log10(x)
def log_scale_inv(logx): return 10**logx
def no_scale(x): return x

nlog_scaler = NewFunctionTransformer(name="nlog_scaler", func=nlog, inverse_func=nlog_inv)
log_scaler = NewFunctionTransformer(name="log_scaler", func=log_scale, inverse_func=log_scale_inv)
no_scaler = NewFunctionTransformer(name="no_scaler", func=no_scale, inverse_func=no_scale)


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
        
        - 'uniform': random uniform sampling
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
# GP prediction gradient functions
#===========================================================

def numerical_kernel_gradient(xs, x_train, gp, h=1e-6):
    """
    Compute numerical gradient of GP kernel k(xs, x_train) with respect to xs.
    
    :param xs: (*array_like, shape (d,) or (1, d)*)
        Query point(s) to compute gradient at
    :param x_train: (*array_like, shape (n, d)*)
        Training points
    :param gp: (*george.GP*)
        Gaussian Process object with kernel
    :param h: (*float, default=1e-6*)
        Step size for finite differences
        
    :returns: *ndarray, shape (n, d)*
        Numerical gradient of kernel k(xs, x_train) w.r.t. xs
        Each row corresponds to gradient w.r.t. one training point
    """
    xs = np.atleast_2d(xs)
    x_train = np.atleast_2d(x_train)
    
    if xs.shape[0] != 1:
        raise ValueError("This function handles single query point only")
    
    n_train, d = x_train.shape
    xs_flat = xs.flatten()
    
    # Initialize gradient array
    grad_k = np.zeros((n_train, d))
    
    # Compute gradient for each dimension
    for i in range(d):
        # Create perturbed points
        xs_plus = xs_flat.copy()
        xs_minus = xs_flat.copy()
        xs_plus[i] += h
        xs_minus[i] -= h
        
        # Evaluate kernel at perturbed points
        k_plus = gp.kernel.get_value(xs_plus.reshape(1, -1), x_train).flatten()
        k_minus = gp.kernel.get_value(xs_minus.reshape(1, -1), x_train).flatten()
        
        # Central difference
        grad_k[:, i] = (k_plus - k_minus) / (2 * h)
    
    return grad_k


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
    
    # Use the grad_gp_kernel function to get kernel gradients
    # grad_ks shape: (n_train_points, n_dims) for single query point

    grad_ks = numerical_kernel_gradient(xs, gp._x, gp, h=1e-6)

    # The GP mean gradient is: ∇μ(x) = ∇k(x, X_train)^T @ α
    # where α = K^{-1} @ y_train
    grad_mean = np.dot(grad_ks.T, gp._alpha)
    
    return grad_mean.flatten()


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
    
    # Get kernel gradients: ∇k(x*, X_train)
    # grad_ks shape: (n_train_points, n_dims) for single query point
    grad_ks = numerical_kernel_gradient(xs, gp._x, gp, h=1e-6)

    # Get kernel values: k(x*, X_train)
    ks = gp.kernel.get_value(xs, gp._x).flatten()  # shape: (n_train_points,)
    
    # Get inverse covariance matrix
    Kinv = gp.solver.get_inverse()  # shape: (n_train_points, n_train_points)
    
    # For stationary kernels, ∇k(x*, x*) = 0, so the variance gradient is:
    # ∇σ²(x*) = -2 * ∇k(x*, X_train)^T @ K^{-1} @ k(x*, X_train)
    # 
    # grad_ks.T shape: (n_dims, n_train_points)
    # Kinv @ ks shape: (n_train_points,)
    # Result shape: (n_dims,)
    
    grad_var = -2.0 * np.dot(grad_ks.T, np.dot(Kinv, ks))
    
    return grad_var.flatten()


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
    
    **References**
    
    Wang & Li (2017): "Adaptive Gaussian Process Approximation for Bayesian 
    Inference with Expensive Likelihood Functions", Neural Computation, 30, 3072-3094.
    """

    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.inf

    mu, var = predict_gp(theta.reshape(1,-1))

    try:
        util = -(mu + 0.5*np.log(2.0*np.pi*np.e*var))
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return util.item()


def grad_agp_utility(theta, gp, bounds):
    
    # Ensure theta is properly shaped
    theta = np.asarray(theta).flatten()
    
    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.full(len(theta), np.inf)  # Return array of infs with correct shape
    
    d_mu = grad_gp_mean_prediction(theta, gp)
    d_var = grad_gp_var_prediction(theta, gp)
    
    # Ensure d_mu and d_var are arrays with correct shape
    d_mu = np.asarray(d_mu).flatten()
    d_var = np.asarray(d_var).flatten()
    
    # Check dimension consistency
    if len(d_mu) != len(theta) or len(d_var) != len(theta):
        raise ValueError(f"Dimension mismatch: theta has {len(theta)} dims, "
                        f"d_mu has {len(d_mu)} dims, d_var has {len(d_var)} dims")

    d_agp = -(d_mu + 0.5 * d_var)

    return d_agp.flatten()  # Ensure output is 1D array


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

    return util.item()


def grad_bape_utility(theta, gp, bounds):
    
    # Ensure theta is properly shaped
    theta = np.asarray(theta).flatten()
    
    if not np.isfinite(lnprior_uniform(theta, bounds)):
        return np.full(len(theta), np.inf)  # Return array of infs with correct shape

    mu, var = gp.predict(gp._y, theta.reshape(1,-1), return_var=True)
    
    d_mu = grad_gp_mean_prediction(theta, gp)
    d_var = grad_gp_var_prediction(theta, gp)
    
    # Check dimension consistency
    if len(d_mu) != len(theta) or len(d_var) != len(theta):
        raise ValueError(f"Dimension mismatch: theta has {len(theta)} dims, "
                        f"d_mu has {len(d_mu)} dims, d_var has {len(d_var)} dims")

    # Gradient of BAPE utility in log form:
    # bape_util = -((2.0 * mu + var) + logsubexp(var, 0.0))
    #           = -(2*mu + var + log(exp(var) - 1))
    # 
    # ∇bape_util/∇μ = -2
    # ∇bape_util/∇σ² = -(1 + exp(σ²)/(exp(σ²) - 1))
    #
    # Using chain rule:
    # ∇bape_util/∇θ = (∇bape_util/∇μ)(∇μ/∇θ) + (∇bape_util/∇σ²)(∇σ²/∇θ)
    
    exp_var = np.exp(var)
    
    # Partial derivatives of the log-form BAPE utility
    d_bape_d_mu = -2.0
    d_bape_d_var = -(1.0 + exp_var / (exp_var - 1.0))
    
    # Apply chain rule
    d_bape = d_bape_d_mu * d_mu + d_bape_d_var * d_var
    
    return d_bape


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

    return util.item()


def assign_utility(algorithm):

    # Assign utility function
    if algorithm == "bape":
        utility = bape_utility
        grad_utility = grad_bape_utility
    elif algorithm == "agp":
        utility = agp_utility
        grad_utility = grad_agp_utility
    elif algorithm == "jones":
        utility = jones_utility
        grad_utility = None
    else:
        print(f"ERROR: Unknown utility function: {algorithm}. Defaulting to BAPE.")
        utility = bape_utility
        grad_utility = grad_bape_utility

    return utility, grad_utility


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
            print("Warning: Acquisition function optimization prior fail", x_opt)
            return np.nan, np.nan
    else:
        print("Warning: Acquisition function optimization infinite fail", x_opt, f_opt)
        return np.nan, np.nan


def minimize_objective(obj_fn, bounds=None, nopt=1, method="l-bfgs-b",
                       ps=None, options=None, grad_obj_fn=None, pool=None):
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
            options = {"maxiter": 100}
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
        grad_obj_fn = None  # Nelder-Mead does not use gradients
    elif str(method).lower() == "l-bfgs-b":
        # Ensure we use limited memory for large problems
        if "maxcor" not in options:
            options["maxcor"] = 10

    # Generate all starting points at once to ensure good space-filling distribution
    # This prevents clustering that can occur when generating points individually
    if ps is None:
        starting_points = prior_sampler(bounds, nsample=nopt, sampler="lhs")
    else:
        # Use provided sampler to generate all points at once
        starting_points = ps(nsample=nopt)
    
    # Pre-flatten starting points to avoid repeated operations
    starting_points = np.array([pt.flatten() for pt in starting_points])
    
    if pool is not None:
        def single_opt(ii):
            return minimize_objective_single(ii, obj_fn, bounds, starting_points[ii], method, options, grad_obj_fn)

        opt_results = pool.imap(single_opt, np.arange(nopt))
        min_theta = [res[0] for res in opt_results]
        min_obj = [res[1] for res in opt_results]
    else:
        # Serial execution 
        min_theta = []
        min_obj = []
        for ii in range(nopt):
            minx, miny = minimize_objective_single(ii, obj_fn, bounds, starting_points[ii], method, options, grad_obj_fn)
            min_theta.append(minx)
            min_obj.append(miny)
  
    # Return value that minimizes objective function out of all minimizations
    # Filter out NaN results
    valid_results = [(theta, obj) for theta, obj in zip(min_theta, min_obj) 
                     if np.all(np.isfinite(theta)) and np.isfinite(obj)]
    
    if len(valid_results) == 0:
        # All optimizations failed - return NaN to signal failure
        print(f"Warning: All {nopt} optimization attempts failed. Returning NaN.")
        return np.nan, np.nan
    
    # Find best among valid results
    valid_theta, valid_obj = zip(*valid_results)
    best_ind = np.argmin(valid_obj)
    theta_best = valid_theta[best_ind]  
    obj_best = valid_obj[best_ind]

    return theta_best, obj_best


class BetaWarpingFunction:
    """
    Swersky et al. 2017 Beta CDF warping function for sklearn FunctionTransformer.
    
    Note: This transformer must be refit when new data extends beyond the 
    original data range to update the internal MinMaxScaler bounds.
    """
    def __init__(self, alpha=2.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta
        self.minmax_scaler = MinMaxScaler()
    
    def fit(self, X, y=None):
        """
        Fit the MinMaxScaler to the data, establishing the data range bounds.
        
        This updates the internal min/max values used for scaling. Should be
        called whenever new data extends beyond the previous range.
        """
        self.minmax_scaler.fit(X)
        return self
    
    def transform(self, X):

        X_scaled = self.minmax_scaler.transform(X)
        
        # Validate that scaled values are in valid range [0, 1]
        if np.any(X_scaled < 0) or np.any(X_scaled > 1):
            invalid_mask = (X_scaled < 0) | (X_scaled > 1)
            invalid_values = X_scaled[invalid_mask]
            raise ValueError(
                f"Scaled values outside [0, 1] range detected: {invalid_values}. "
                f"Min={np.min(X_scaled)}, Max={np.max(X_scaled)}. "
                f"This indicates the transformer needs to be refit with the new data range. "
                f"Call fit() with data that includes both old and new samples."
            )
        
        # Apply Beta CDF warping per feature
        X_warped = np.zeros_like(X_scaled)
        for i in range(X_scaled.shape[1]):
            X_warped[:, i] = betainc(self.alpha, self.beta, X_scaled[:, i])
        
        return X_warped
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        
        # Validate input range
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError(
                f"Input to inverse_transform must be in [0, 1]. "
                f"Got min={np.min(X)}, max={np.max(X)}"
            )
        
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
        check_inverse=False
    )