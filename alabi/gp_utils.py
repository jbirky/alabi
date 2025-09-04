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

__all__ = ["configure_gp", 
           "optimize_gp",
           "grad_gp_mean_prediction",
           "grad_gp_var_prediction",
           "grad_gp_kernel"]


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


def grad_gp_kernel(x1, x2, gp, wrt='x1'):
    """
    Compute the gradient of the GP kernel k(x1, x2) with respect to x1 or x2.
    
    For a GP kernel k(x1, x2), this function computes:
    - If wrt='x1': ∇_{x1} k(x1, x2) 
    - If wrt='x2': ∇_{x2} k(x1, x2)
    
    This is useful for computing derivatives of covariance functions and
    can be used in active learning acquisition functions that require
    kernel gradients.
    
    Parameters
    ----------
    x1 : array_like, shape (n1, d) or (d,)
        First input points. If 1D, will be reshaped to (1, d).
    x2 : array_like, shape (n2, d) or (d,)  
        Second input points. If 1D, will be reshaped to (1, d).
    gp : george.GP
        Fitted George Gaussian Process object with kernel
    wrt : str, optional
        Variable to take gradient with respect to. Either 'x1' or 'x2'.
        Default is 'x1'.
        
    Returns
    -------
    grad_k : ndarray, shape (n2, d) for single x1, or (n1, n2, d) for multiple x1
        Gradient of kernel k(x1, x2) with respect to the specified variable.
        
    Notes
    -----
    This function uses George's built-in gradient capabilities which are
    analytically computed for supported kernels (ExpSquared, Matern, etc.).
    
    For kernel k(x1, x2), the gradient with respect to x1 is typically:
    ∇_{x1} k(x1, x2) = -∇_{x2} k(x1, x2)
    
    Examples
    --------
    >>> import numpy as np
    >>> from george import kernels, GP
    >>> 
    >>> # Create GP with ExpSquared kernel
    >>> kernel = kernels.ExpSquaredKernel(1.0, ndim=2)
    >>> gp = GP(kernel)
    >>> 
    >>> # Compute kernel gradient
    >>> x1 = np.array([[0.0, 0.0]])  # shape (1, 2)
    >>> x2 = np.array([[1.0, 0.5], [0.5, 1.0]])  # shape (2, 2)
    >>> grad_k = grad_gp_kernel(x1, x2, gp, wrt='x1')
    >>> print(grad_k.shape)  # (2, 2) - gradient for each x2 point
    """
    
    # Ensure inputs are properly shaped
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    if x1.shape[1] != x2.shape[1]:
        raise ValueError(f"x1 and x2 must have same number of dimensions. "
                        f"Got x1: {x1.shape[1]}, x2: {x2.shape[1]}")
    
    if wrt not in ['x1', 'x2']:
        raise ValueError(f"wrt must be 'x1' or 'x2', got {wrt}")
    
    # Get kernel gradients using George's built-in methods
    if wrt == 'x1':
        # Gradient with respect to first argument
        # George's get_gradient computes ∇_{x1} k(x1, x2)
        grad_k = gp.kernel.get_gradient(x1, x2)
    else:
        # Gradient with respect to second argument  
        # For symmetric kernels: ∇_{x2} k(x1, x2) = -∇_{x1} k(x2, x1)
        grad_k = -gp.kernel.get_gradient(x2, x1)
        # Need to transpose to get correct shape
        if grad_k.ndim == 3:
            grad_k = grad_k.transpose(1, 0, 2)
    
    # Extract only spatial gradients (first ndim components)
    # George's gradient includes both spatial and hyperparameter gradients
    ndim = x1.shape[1]
    
    # Get the spatial part of the gradient
    if grad_k.ndim == 3:
        grad_k_spatial = grad_k[:, :, :ndim]
    else:
        grad_k_spatial = grad_k[:, :ndim]
    
    # For single x1 point, return shape (n2, d)
    if x1.shape[0] == 1:
        if grad_k_spatial.ndim == 3:
            return grad_k_spatial[0]  # shape (n2, d)
        else:
            return grad_k_spatial  # already (n2, d)
    else:
        # Multiple x1 points - return full gradient array (n1, n2, d)
        return grad_k_spatial


def numerical_kernel_gradient(xs, x_train, gp, h=1e-6):
    """
    Compute numerical gradient of GP kernel k(xs, x_train) with respect to xs.
    
    Parameters:
    -----------
    xs : array_like, shape (d,) or (1, d)
        Query point(s) to compute gradient at
    x_train : array_like, shape (n, d) 
        Training points
    gp : george.GP
        Gaussian Process object with kernel
    h : float, default=1e-6
        Step size for finite differences
        
    Returns:
    --------
    grad_k : ndarray, shape (n, d)
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
    
    # Ensure GP is computed
    if not hasattr(gp, '_alpha') or gp._alpha is None:
        raise ValueError("GP must be computed before computing gradients")
    
    # Use the grad_gp_kernel function to get kernel gradients
    # grad_ks shape: (n_train_points, n_dims) for single query point
    # grad_ks = grad_gp_kernel(xs, gp._x, gp, wrt='x1')

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
    
    # Ensure GP is computed
    if not hasattr(gp, '_alpha') or gp._alpha is None:
        raise ValueError("GP must be computed before computing gradients")
    
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


def optimize_gp(gp, theta, y, gp_hyper_prior, p0, bounds=None,
                method="l-bfgs-b", optimizer_kwargs=None, max_iter=50):

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Collapse arrays if 1D
    theta = theta.squeeze()
    y = y.squeeze()

    # initial hyperparameters
    init_hp = gp.get_parameter_vector()
        
    valid_methods = ["newton-cg", "bfgs", "l-bfgs-b", "powell"]
    if method not in valid_methods:
        print(f"Warning: {method} not in valid methods {valid_methods}. Using 'l-bfgs-b' optimizer instead.")
        method = "l-bfgs-b"
        
    # methods without bounds arg 
    if method in ["bfgs"]:
        bounds = None

    # Minimize GP nll, save result, evaluate marginal likelihood
    if method in ["newton-cg", "l-bfgs-b"]:
        jac = _grad_nll
    else:
        jac = None
        
    # Set improved default optimizer_kwargs for faster convergence
    if optimizer_kwargs is None:
        default_optimizer_kwargs = {
            'newton-cg': {'maxiter': max_iter},
            'bfgs': {'maxiter': max_iter},
            'l-bfgs-b': {'maxiter': max_iter, 'factr': 1e12},
            'powell': {'maxiter': max_iter},
        }
        optimizer_kwargs = default_optimizer_kwargs.get(method.lower(), {})
    
    nopt = p0.shape[0] if p0.ndim > 1 else 1
    if nopt > 1:
        # Run the optimization routine nopt times
        res = []
        mll = []
        
        for ii, x0 in enumerate(p0):
            try:
                result = minimize(_nll, x0, args=(gp, y, gp_hyper_prior), method=method,
                                jac=jac, bounds=bounds, options=optimizer_kwargs)
                
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