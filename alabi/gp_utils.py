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
           "numerical_kernel_gradient"]


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
    """
    Configure and initialize a Gaussian Process with robust error handling.
    
    Creates a george.GP object with the specified kernel and configuration options.
    Includes automatic fixes for common numerical issues such as singular matrices,
    duplicate points, and poor conditioning.
    
    :param theta: (*array-like, shape (n_samples, n_features)*)
        Training input locations (parameters). Must contain finite values only.
        
    :param y: (*array-like, shape (n_samples,)*)
        Training target values (function evaluations). Must contain finite values only.
        
    :param kernel: (*george kernel object*)
        George kernel object defining the covariance function. Common options include
        ExpSquaredKernel, Matern32Kernel, Matern52Kernel.
        
    :param fit_amp: (*bool, optional, default=True*)
        Whether to fit the kernel amplitude. If True, scales the kernel by the 
        variance of y to improve conditioning.
        
    :param fit_mean: (*bool, optional, default=True*)
        Whether to fit a constant mean function. If True, initializes mean to 
        median(y) and allows optimization.
        
    :param fit_white_noise: (*bool, optional, default=False*)
        Whether to fit the white noise (nugget) parameter. If True, the noise
        level will be optimized during hyperparameter tuning.
        
    :param white_noise: (*float, optional, default=-12*)
        Log-scale white noise parameter. Acts as regularization to prevent
        singular matrices. More negative values = less noise.
        
    :param hyperparameters: (*array-like, optional, default=None*)
        Pre-specified hyperparameters to set. If provided, these values are used
        instead of the kernel's default initialization.
        
    :returns: *george.GP or None*
        Configured and computed GP object ready for predictions, or None if
        configuration failed despite all attempted fixes.
        
    :raises ValueError:
        If theta or y contain non-finite values (NaN or inf).
        
    :raises LinAlgError:
        If GP computation fails due to singular covariance matrix, automatically
        attempts several fixes before giving up.
        
    **Notes**
    
    The function includes several automatic fixes for numerical issues:
    
    1. **Jitter addition**: Progressively increases white noise to regularize
    2. **Duplicate removal**: Detects and removes duplicate training points
    3. **Error reporting**: Provides diagnostic information for debugging
    """

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
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError during GP computation: {e}")
        print(f"This usually indicates numerical issues. Trying fixes:")
        
        # Fix 1: Add more white noise (jitter) to diagonal
        original_white_noise = white_noise
        for jitter_level in [-10, -8, -6, -4]:
            try:
                print(f"  Trying with white_noise={jitter_level}")
                gp = george.GP(kernel=kernel, fit_mean=fit_mean, mean=np.median(y),
                              white_noise=jitter_level, fit_white_noise=fit_white_noise)
                if hyperparameters is not None:
                    gp.set_parameter_vector(hyperparameters)
                gp.compute(theta)
                print(f"  ✓ Success with white_noise={jitter_level}")
                break
            except np.linalg.LinAlgError:
                continue
        else:
            # Fix 2: Check for duplicate points
            unique_theta, unique_indices = np.unique(theta, axis=0, return_index=True)
            if len(unique_theta) < len(theta):
                print(f"  Found {len(theta) - len(unique_theta)} duplicate points, removing them")
                theta = unique_theta
                y = y[unique_indices]
                try:
                    gp = george.GP(kernel=kernel, fit_mean=fit_mean, mean=np.median(y),
                                  white_noise=original_white_noise, fit_white_noise=fit_white_noise)
                    if hyperparameters is not None:
                        gp.set_parameter_vector(hyperparameters)
                    gp.compute(theta)
                    print(f"  ✓ Success after removing duplicates")
                except np.linalg.LinAlgError:
                    print(f"  Still failed after removing duplicates")
                    return None
            else:
                print(f"  All fixes failed. This might indicate:")
                print(f"    - Inappropriate kernel parameters")
                print(f"    - Poorly scaled data")
                print(f"    - Pathological likelihood surface")
                return None
    except Exception as e:
        print(f"Other error during GP computation: {e}")
        return None

    return gp


def optimize_gp(gp, theta, y, gp_hyper_prior, p0, bounds=None,
                method="l-bfgs-b", optimizer_kwargs=None, max_iter=50):
    """
    Optimize Gaussian Process hyperparameters by maximizing marginal likelihood.
    
    Performs hyperparameter optimization for a Gaussian Process using scipy's
    minimize function. Supports multiple optimization restarts and automatically
    selects the result with highest marginal likelihood.
    
    :param gp: (*george.GP*)
        Configured Gaussian Process object. Should be computed with training data.
        
    :param theta: (*array-like, shape (n_samples, n_features)*)
        Training input locations (parameters). Will be squeezed if 1D.
        
    :param y: (*array-like, shape (n_samples,)*)
        Training target values (function evaluations). Will be squeezed if 1D.
        
    :param gp_hyper_prior: (*callable*)
        Prior function for hyperparameters. Should return log-probability density
        for given hyperparameter vector. Used to constrain optimization.
        
    :param p0: (*array-like, shape (n_restarts, n_hyperparams) or (n_hyperparams,)*)
        Initial guesses for hyperparameter optimization. If 2D, performs multiple
        restarts with different initializations.
        
    :param bounds: (*list of tuples, optional, default=None*)
        Bounds for hyperparameter optimization as [(min, max), ...]. Only used
        for methods that support bounds (e.g., 'l-bfgs-b').
        
    :param method: (*str, optional, default="l-bfgs-b"*)
        Scipy optimization method. Supported methods:
        
        - 'l-bfgs-b': L-BFGS-B with bounds support (default)
        - 'newton-cg': Newton-CG with gradients
        - 'bfgs': BFGS (no bounds support)
        - 'powell': Powell method (derivative-free)
        
    :param optimizer_kwargs: (*dict, optional, default=None*)
        Additional keyword arguments passed to scipy.optimize.minimize.
        If None, uses method-specific defaults optimized for GP optimization.
        
    :param max_iter: (*int, optional, default=50*)
        Maximum number of iterations for optimization. Used as default in
        optimizer_kwargs if not specified.
        
    :returns: *george.GP or None*
        GP object with optimized hyperparameters, or None if optimization failed.
        The returned GP is recomputed with optimal hyperparameters.
        
    **Notes**
    
    **Multiple Restarts**: If p0 is 2D, the function performs multiple optimization
    restarts and selects the result with highest marginal log-likelihood. This helps
    avoid local minima in the hyperparameter space.
    
    **Gradient Usage**: For methods 'newton-cg' and 'l-bfgs-b', analytical gradients
    are used via the `_grad_nll` function for faster convergence.
    
    **Error Handling**: The function gracefully handles optimization failures and
    falls back to initial hyperparameters when necessary.
    
    **Examples**
    
    Basic hyperparameter optimization:
    
    .. code-block:: python
    
        import numpy as np
        from scipy.stats import uniform
        
        # Define prior function
        def prior(params):
            # Uniform prior on log-scale parameters
            if np.all(params > -10) and np.all(params < 10):
                return 0.0  # log(1.0)
            return -np.inf
        
        # Single optimization
        p0 = gp.get_parameter_vector()  # Current hyperparameters
        optimized_gp = optimize_gp(gp, theta, y, prior, p0)
    
    Multiple restarts for robustness:
    
    .. code-block:: python
    
        # Multiple random initializations
        n_restarts = 5
        n_params = len(gp.get_parameter_vector())
        p0_multi = np.random.randn(n_restarts, n_params)
        
        # Optimization with bounds
        bounds = [(-5, 5)] * n_params  # Bound each parameter
        
        optimized_gp = optimize_gp(gp, theta, y, prior, p0_multi, 
                                  bounds=bounds, method='l-bfgs-b')
    
    Custom optimization settings:
    
    .. code-block:: python
    
        # High-precision optimization
        custom_opts = {
            'maxiter': 200,
            'ftol': 1e-12,
            'gtol': 1e-8
        }
        
        optimized_gp = optimize_gp(gp, theta, y, prior, p0,
                                  method='l-bfgs-b',
                                  optimizer_kwargs=custom_opts)
    """

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
                       jac=jac, bounds=bounds, options=optimizer_kwargs)

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