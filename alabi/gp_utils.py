"""
:py:mod:`gp_utils.py` 
-------------------------------------------------

Gaussian process utility functions for initializing GPs and optimizing their
hyperparameters.

"""

import alabi
import numpy as np
import george
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from functools import partial
import copy
import warnings
from tqdm import tqdm

__all__ = ["configure_gp", 
           "optimize_gp",
           "optimize_gp_kfold_cv",
           "weighted_mse_by_probability",
           "grad_gp_mean_prediction",
           "grad_gp_var_prediction",
           "numerical_kernel_gradient"]


def weighted_mse_by_probability(y_true, y_pred, weight_method='exponential', temperature=1.0):
    """
    Compute weighted MSE where higher probability values get larger weights.
    
    Parameters:
    -----------
    y_true : array-like
        True log likelihood values (more negative = lower probability)
    y_pred : array-like  
        Predicted log likelihood values
    weight_method : str
        Method for computing weights from log likelihood:
        - 'exponential': w = exp(y_true / temperature) 
        - 'linear': w = y_true - min(y_true) + epsilon
        - 'softmax': w = softmax(y_true / temperature)
        - 'rank': w based on rank order of y_true
    temperature : float
        Temperature parameter for exponential/softmax weighting
        
    Returns:
    --------
    weighted_mse : float
        Weighted mean squared error
    weights : array
        The weights used for each point
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if weight_method == 'exponential':
        # Higher log likelihood = higher probability = larger weight
        weights = np.exp(y_true / temperature)
        
    elif weight_method == 'linear':
        # Shift to make all weights positive, higher y_true gets higher weight
        epsilon = 1e-6
        weights = y_true - np.min(y_true) + epsilon
        
    elif weight_method == 'softmax':
        # Softmax weighting - probabilistic approach
        weights = np.exp(y_true / temperature)
        weights = weights / np.sum(weights) * len(weights)  # Normalize to maintain scale
        
    elif weight_method == 'rank':
        # Rank-based weighting - highest log likelihood gets highest weight
        ranks = np.argsort(np.argsort(y_true))  # Get ranks (0 to n-1)
        weights = ranks + 1  # Make weights 1 to n
        
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")
    
    # Normalize weights to have mean = 1 (maintains MSE scale)
    weights = weights / np.mean(weights)
    
    # Compute weighted MSE
    mse_values = (y_true - y_pred) ** 2
    weighted_mse = np.average(mse_values, weights=weights)
    
    return weighted_mse


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


def optimize_gp(gp, _theta, _y, gp_hyper_prior, p0, bounds=None,
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
    _theta = _theta.squeeze()
    _y = _y.squeeze()

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
                result = minimize(_nll, x0, args=(gp, _y, gp_hyper_prior), method=method,
                                jac=jac, bounds=bounds, options=optimizer_kwargs)
                
                if result.success and np.isfinite(gp_hyper_prior(result.x)):
                    # Compute marginal log likelihood for this set of kernel hyperparameters
                    test_gp = copy.copy(gp)
                    test_gp.set_parameter_vector(result.x)
                    test_gp.recompute()
                    current_mll = test_gp.log_likelihood(_y, quiet=True)
                    
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
        res = minimize(_nll, p0, args=(gp, _y, gp_hyper_prior), method=method,
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


def _evaluate_candidate_worker(args):
    """
    Worker function for parallel evaluation of a single hyperparameter candidate.
    
    :param args: tuple containing (cand_idx, hyperparams, gp, theta, y,  
                                   k_folds, scoring, random_state, weighted_mse_method, weighted_mse_temperature)
    :returns: tuple (cand_idx, fold_scores, success_flag)
    """
    import copy
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    (cand_idx, hyperparams, gp, _theta, _y, y_scaler, k_folds, scoring, random_state, weighted_mse_method, weighted_mse_temperature) = args

    try:
        # Check for basic hyperparameter validity
        if np.any(np.isnan(hyperparams)) or np.any(np.isinf(hyperparams)):
            return (cand_idx, None, "Invalid hyperparameters (NaN/Inf)")
        
        # Set up cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        fold_scores = []
        fold_idx = 0
        
        # Perform k-fold cross validation
        for train_idx, val_idx in kf.split(_theta):
            try:
                # Split data
                _theta_train, _theta_val = _theta[train_idx], _theta[val_idx]
                _y_train, _y_val = _y[train_idx], _y[val_idx]
                
                # Check training data validity
                if np.any(np.isnan(_y_train)) or np.any(np.isinf(_y_train)):
                    raise ValueError(f"Training targets contain NaN/Inf values")
                if np.any(np.isnan(_theta_train)) or np.any(np.isinf(_theta_train)):
                    raise ValueError(f"Training inputs contain NaN/Inf values")
                
                # Ensure proper shapes for george GP
                _theta_train = np.atleast_2d(_theta_train)
                _theta_val = np.atleast_2d(_theta_val)
                
                # Handle case where we have single samples
                if _theta_train.shape[0] == 1 and _theta_train.shape[1] != _theta.shape[1]:
                    _theta_train = _theta_train.T
                if _theta_val.shape[0] == 1 and _theta_val.shape[1] != _theta.shape[1]:
                    _theta_val = _theta_val.T
                
                # Create and configure GP for this fold
                gp_fold = copy.deepcopy(gp)
                
                # Check for invalid hyperparameters before setting
                if np.any(np.isnan(hyperparams)) or np.any(np.isinf(hyperparams)):
                    raise ValueError(f"Invalid hyperparameters contain NaN/Inf: {hyperparams}")
                
                gp_fold.set_parameter_vector(hyperparams)
                gp_fold.compute(_theta_train)
                
                # Check if GP computation was successful
                current_hp = gp_fold.get_parameter_vector()
                if np.any(np.isnan(current_hp)) or np.any(np.isinf(current_hp)):
                    raise ValueError(f"GP parameters became invalid after compute: {current_hp}")
                
                # Check GP log-likelihood for numerical stability
                try:
                    log_like = gp_fold.log_likelihood(_y_train)
                    if np.isnan(log_like) or np.isinf(log_like):
                        raise ValueError(f"GP log-likelihood is invalid: {log_like} with hyperparams: {hyperparams}")
                except Exception as e:
                    raise ValueError(f"Failed to compute GP log-likelihood: {str(e)} with hyperparams: {hyperparams}")
                
                # Make predictions on validation set
                try:
                    _y_pred = gp_fold.predict(_y_train, _theta_val, return_var=False, return_cov=False)
                except Exception as e:
                    raise ValueError(f"GP prediction failed: {str(e)} with hyperparams: {hyperparams} "
                                   f"train_shape: {_theta_train.shape}, val_shape: {_theta_val.shape}")
                
                # Check for NaN or infinite values
                if len(_y_pred) == 0:
                    raise ValueError(f"GP predictions are empty (no predictions returned)")
                elif np.any(np.isnan(_y_pred)) or np.any(np.isinf(_y_pred)):
                    mean_val = np.nanmean(_y_pred) if len(_y_pred) > 0 else float('nan')
                    std_val = np.nanstd(_y_pred) if len(_y_pred) > 0 else float('nan')
                    raise ValueError(f"GP predictions contain NaN or Inf values: _y_pred stats: "
                                   f"mean={mean_val}, std={std_val}, "
                                   f"n_nan={np.sum(np.isnan(_y_pred))}, n_inf={np.sum(np.isinf(_y_pred))}")
                
                if np.any(np.isnan(_y_val)) or np.any(np.isinf(_y_val)):
                    raise ValueError(f"Validation targets contain NaN or Inf values")
                
                # Compute error using unscaled y values 
                y_val = y_scaler.inverse_transform(_y_val.reshape(-1, 1)).flatten()
                y_pred = y_scaler.inverse_transform(_y_pred.reshape(-1, 1)).flatten()
                
                # Calculate fold score
                if scoring == 'mse':
                    fold_score = mean_squared_error(y_val, y_pred)
                elif scoring == 'mae':
                    fold_score = mean_absolute_error(y_val, y_pred)
                elif scoring == 'r2':
                    fold_score = -r2_score(y_val, y_pred)  # Negative because we minimize
                elif scoring == 'weighted_mse':
                    fold_score = weighted_mse_by_probability(y_val, y_pred, 
                                                           weight_method=weighted_mse_method,
                                                           temperature=weighted_mse_temperature)
                else:
                    raise ValueError(f"Unsupported scoring method: {scoring}")
                
                fold_scores.append(fold_score)
                
            except Exception as e:
                fold_scores.append(np.inf)
            
            fold_idx += 1
        
        return (cand_idx, fold_scores, "success")
        
    except Exception as e:
        return (cand_idx, None, str(e))


def optimize_gp_kfold_cv(gp, _theta, _y, hyperparameter_candidates, y_scaler,
                         k_folds=5, scoring='mse', random_state=42, pool=None, 
                         two_stage=False, stage2_candidates=None, stage2_width=0.5,
                         weighted_mse_method='exponential', weighted_mse_temperature=1.0,
                         verbose=True):
    """
    Optimize Gaussian Process hyperparameters using k-fold cross-validation.
    
    This function evaluates different hyperparameter configurations using k-fold 
    cross-validation to select the configuration that generalizes best to unseen data.
    This approach helps prevent overfitting compared to standard marginal likelihood
    maximization.
    
    :param gp: (*george.GP*)
        Configured Gaussian Process object template. Will be copied for each CV fold.
        
    :param _theta: (*array-like, shape (n_samples, n_features)*)
        Training input locations (parameters). Must have at least k_folds samples.
        
    :param _y: (*array-like, shape (n_samples,)*)
        Training target values (function evaluations). Must match _theta length.
        
    :param gp_hyper_prior: (*callable*)
        Prior function for hyperparameters. Should return log-probability density
        for given hyperparameter vector. Used to constrain search space.
        
    :param hyperparameter_candidates: (*array-like, shape (n_candidates, n_hyperparams)*)
        Array of hyperparameter vectors to evaluate via cross-validation.
        Each row represents one hyperparameter configuration to test.
        
    :param k_folds: (*int, optional, default=5*)
        Number of folds for cross-validation. Must be >= 2 and <= n_samples.
        Common choices: 5 or 10 for good bias-variance tradeoff.
        
    :param scoring: (*str, optional, default='mse'*)
        Scoring metric for cross-validation. Supported options:
        
        - 'mse': Mean Squared Error (lower is better)
        - 'mae': Mean Absolute Error (lower is better)  
        - 'r2': R-squared coefficient (higher is better)
        - 'weighted_mse': Weighted MSE giving higher weight to high-probability regions
        
    :param random_state: (*int, optional, default=42*)
        Random seed for reproducible cross-validation splits.
        
    :param pool: (*multiprocessing.Pool, optional, default=None*)
        Multiprocessing pool for parallel evaluation of hyperparameter candidates.
        If None, evaluation runs sequentially. If provided, candidates are 
        evaluated in parallel using the pool's worker processes.
        
    :param two_stage: (*bool, optional, default=False*)
        Whether to use two-stage optimization (explore-exploit strategy):
        
        - Stage 1: Random exploration with provided candidates (explore)
        - Stage 2: Focused grid search around best result (exploit)
        
    :param stage2_candidates: (*int, optional, default=None*)
        Number of candidates for stage 2 grid search. If None, uses 
        len(hyperparameter_candidates) // 2.
        
    :param stage2_width: (*float, optional, default=0.5*)
        Width factor for stage 2 search around best parameters.
        Smaller values = tighter search, larger values = wider search.
        
    :param weighted_mse_method: (*str, optional, default='exponential'*)
        Weighting method for weighted_mse scoring. Options:
        - 'exponential': w = exp(y_true / temperature)
        - 'linear': w = y_true - min(y_true) + epsilon  
        - 'softmax': w = softmax(y_true / temperature)
        - 'rank': w based on rank order of y_true
        
    :param weighted_mse_temperature: (*float, optional, default=1.0*)
        Temperature parameter for exponential/softmax weighting.
        Lower values emphasize high-probability regions more strongly.
        
    :returns:
        - **best_gp** (*george.GP*) -- GP with optimal hyperparameters set
        - **best_hyperparams** (*array*) -- Best hyperparameter vector
        - **cv_results** (*dict*) -- Detailed CV results with scores and statistics
        
    :raises:
        - **ValueError** -- If insufficient data, invalid parameters, or no valid hyperparameters
        - **RuntimeError** -- If all hyperparameter evaluations fail
        
    Example:
        >>> # Define candidate hyperparameters
        >>> candidates = np.array([
        ...     [0.1, 1.0, -5.0],  # [log_amplitude, log_length_scale, log_noise]
        ...     [0.5, 0.5, -4.0],
        ...     [1.0, 2.0, -6.0]
        ... ])
        >>> 
        >>> # Optimize using 5-fold CV
        >>> best_gp, best_params, results = optimize_gp_kfold_cv(
        ...     gp, theta, y, prior_fn, candidates, k_folds=5
        ... )
        >>> print(f"Best hyperparameters: {best_params}")
        >>> print(f"CV score: {results['best_score']:.4f} ± {results['best_score_std']:.4f}")
    """
    
    # Input validation
    _theta = np.asarray(_theta)
    _y = np.asarray(_y)
    hyperparameter_candidates = np.asarray(hyperparameter_candidates)

    if _theta.ndim == 1:
        _theta = _theta.reshape(-1, 1)
    if _y.ndim != 1:
        _y = _y.squeeze()
        
    n_samples = len(_theta)
    if len(_y) != n_samples:
        raise ValueError(f"_theta and _y must have same length, got {len(_theta)} and {len(_y)}")

    if n_samples < k_folds:
        raise ValueError(f"Number of samples ({n_samples}) must be >= k_folds ({k_folds})")
        
    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}")
        
    if hyperparameter_candidates.ndim == 1:
        hyperparameter_candidates = hyperparameter_candidates.reshape(1, -1)
        
    n_candidates, n_hyperparams = hyperparameter_candidates.shape
    
    # Set up cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    # Storage for results
    cv_scores = np.zeros((n_candidates, k_folds))
    cv_scores.fill(np.inf)  # Initialize with worst possible score
    
    # Two-stage optimization: explore then exploit
    if two_stage and verbose:
        print("=== STAGE 1: EXPLORATION (Random Search) ===")
    
    print(f"Evaluating {n_candidates} hyperparameter candidates using {k_folds}-fold CV...")
    
    if pool is not None:
        # Parallel evaluation using multiprocessing pool
        print(f"Using multiprocessing pool with {pool._processes} processes")
        
        # Prepare arguments for all candidates
        worker_args = []
        for cand_idx, hyperparams in enumerate(hyperparameter_candidates):
            args = (cand_idx, hyperparams, gp, _theta, _y, y_scaler, 
                   k_folds, scoring, random_state, weighted_mse_method, weighted_mse_temperature)
            worker_args.append(args)
        
        # Evaluate all candidates in parallel with progress bar
        with tqdm(total=len(worker_args), desc="Evaluating candidates", unit="candidate") as pbar:
            # Use pool.imap for progress tracking
            results = []
            for result in pool.imap(_evaluate_candidate_worker, worker_args):
                results.append(result)
                pbar.update(1)
        
        # Process results
        for cand_idx, fold_scores, status in results:
            if status == "success" and fold_scores is not None:
                # Store fold scores
                for fold_idx, score in enumerate(fold_scores):
                    cv_scores[cand_idx, fold_idx] = score
                
                # Calculate and print statistics
                valid_scores = [s for s in fold_scores if np.isfinite(s)]
                if len(valid_scores) > 0:
                    mean_score = np.mean(valid_scores)
                    std_score = np.std(valid_scores)
                    if verbose:
                        print(f"  Candidate {cand_idx+1}/{n_candidates}: {scoring}={mean_score:.4f}±{std_score:.4f} "
                            f"({len(valid_scores)}/{k_folds} folds successful)")
                else:
                    if verbose:
                        print(f"  Candidate {cand_idx+1}/{n_candidates}: All folds failed")
            else:
                # Candidate failed completely
                if verbose:
                    print(f"  Candidate {cand_idx+1}/{n_candidates}: {status}")
                    if "hyperparams" in status.lower() or "prior" in status.lower():
                        print(f"    Hyperparams: {hyperparameter_candidates[cand_idx]}")
    
    else:
        # Sequential evaluation using the worker function
        for cand_idx, hyperparams in enumerate(hyperparameter_candidates):
            args = (cand_idx, hyperparams, gp, _theta, _y, y_scaler, k_folds, scoring, random_state, weighted_mse_method, weighted_mse_temperature)
            results = _evaluate_candidate_worker(args)
            cand_idx_result, fold_scores, status = results
            
            if status == "success" and fold_scores is not None:
                # Store fold scores
                for fold_idx, score in enumerate(fold_scores):
                    cv_scores[cand_idx, fold_idx] = score
                
                # Calculate and print statistics
                valid_scores = [s for s in fold_scores if np.isfinite(s)]
                if len(valid_scores) > 0:
                    mean_score = np.mean(valid_scores)
                    std_score = np.std(valid_scores)
                    if verbose:
                        print(f"  Candidate {cand_idx+1}/{n_candidates}: {scoring}={mean_score:.4f}±{std_score:.4f} "
                            f"({len(valid_scores)}/{k_folds} folds successful)")
                else:
                    if verbose:
                        print(f"  Candidate {cand_idx+1}/{n_candidates}: All folds failed")
            else:
                # Candidate failed completely
                if verbose:
                    print(f"  Candidate {cand_idx+1}/{n_candidates}: {status}")
                    if "hyperparams" in status.lower() or "prior" in status.lower():
                        print(f"    Hyperparams: {hyperparams}")
    
    # Find best hyperparameters
    # Calculate mean CV score for each candidate (only over successful folds)
    mean_cv_scores = []
    std_cv_scores = []
    
    for cand_idx in range(n_candidates):
        fold_scores = cv_scores[cand_idx, :]
        valid_scores = fold_scores[np.isfinite(fold_scores)]
        
        if len(valid_scores) > 0:
            mean_cv_scores.append(np.mean(valid_scores))
            std_cv_scores.append(np.std(valid_scores))
        else:
            mean_cv_scores.append(np.inf)
            std_cv_scores.append(np.inf)
    
    mean_cv_scores = np.array(mean_cv_scores)
    std_cv_scores = np.array(std_cv_scores)
    
    # Find best candidate (lowest score for most metrics, highest for R²)
    if not np.all(np.isinf(mean_cv_scores)):
        best_idx = np.argmin(mean_cv_scores)
        best_hyperparams = hyperparameter_candidates[best_idx]
        best_score = mean_cv_scores[best_idx]
        best_score_std = std_cv_scores[best_idx]
        
        if verbose:
            print(f"\nStage 1 best hyperparameters (candidate {best_idx+1}): {best_hyperparams}")
            print(f"Stage 1 best CV {scoring}: {best_score:.4f} ± {best_score_std:.4f}")
        
        # Store stage 1 results
        stage1_results = {
            'best_score': best_score,
            'best_score_std': best_score_std,
            'best_hyperparams': best_hyperparams,
            'best_candidate_idx': best_idx,
            'all_scores': mean_cv_scores,
            'all_scores_std': std_cv_scores,
            'cv_scores_matrix': cv_scores
        }
        
        # === STAGE 2: EXPLOITATION (Grid Search around best) ===
        if two_stage:
            if verbose:
                print(f"\n=== STAGE 2: EXPLOITATION (Grid Search around best) ===")
            
            # Determine number of stage 2 candidates
            if stage2_candidates is None:
                stage2_candidates = max(n_candidates // 2, 5)
            
            # Generate grid around best hyperparameters
            stage2_hyperparam_candidates = _generate_stage2_candidates(
                best_hyperparams, stage2_candidates, stage2_width, random_state
            )
            
            # Evaluate stage 2 candidates
            n_candidates_s2 = len(stage2_hyperparam_candidates)
            cv_scores_s2 = np.zeros((n_candidates_s2, k_folds))
            cv_scores_s2.fill(np.inf)
            
            if verbose:
                print(f"Evaluating {n_candidates_s2} stage 2 candidates using {k_folds}-fold CV...")
            
            if pool is not None:
                # Parallel evaluation for stage 2
                worker_args_s2 = []
                for cand_idx, hyperparams in enumerate(stage2_hyperparam_candidates):
                    args = (cand_idx, hyperparams, gp, _theta, _y, y_scaler,
                           k_folds, scoring, random_state, weighted_mse_method, weighted_mse_temperature)
                    worker_args_s2.append(args)
                
                with tqdm(total=len(worker_args_s2), desc="Stage 2 candidates", unit="candidate") as pbar:
                    results_s2 = []
                    for result in pool.imap(_evaluate_candidate_worker, worker_args_s2):
                        results_s2.append(result)
                        pbar.update(1)
                
                # Process stage 2 results
                for cand_idx, fold_scores, status in results_s2:
                    if status == "success" and fold_scores is not None:
                        for fold_idx, score in enumerate(fold_scores):
                            cv_scores_s2[cand_idx, fold_idx] = score
                        
                        valid_scores = [s for s in fold_scores if np.isfinite(s)]
                        if len(valid_scores) > 0:
                            mean_score = np.mean(valid_scores)
                            std_score = np.std(valid_scores)
                            if verbose:
                                print(f"  Stage 2 Candidate {cand_idx+1}/{n_candidates_s2}: {scoring}={mean_score:.4f}±{std_score:.4f} "
                                    f"({len(valid_scores)}/{k_folds} folds successful)")
                        else:
                            if verbose:
                                print(f"  Stage 2 Candidate {cand_idx+1}/{n_candidates_s2}: All folds failed")
                    else:
                        if verbose:
                            print(f"  Stage 2 Candidate {cand_idx+1}/{n_candidates_s2}: {status}")
            
            else:
                # Sequential evaluation for stage 2
                for cand_idx, hyperparams in enumerate(stage2_hyperparam_candidates):
                    args = (cand_idx, hyperparams, gp, _theta, _y, y_scaler, k_folds, scoring, random_state, weighted_mse_method, weighted_mse_temperature)
                    results_s2 = _evaluate_candidate_worker(args)
                    cand_idx_result, fold_scores, status = results_s2
                    
                    if status == "success" and fold_scores is not None:
                        for fold_idx, score in enumerate(fold_scores):
                            cv_scores_s2[cand_idx, fold_idx] = score
                        
                        valid_scores = [s for s in fold_scores if np.isfinite(s)]
                        if len(valid_scores) > 0:
                            mean_score = np.mean(valid_scores)
                            std_score = np.std(valid_scores)
                            if verbose:
                                print(f"  Stage 2 Candidate {cand_idx+1}/{n_candidates_s2}: {scoring}={mean_score:.4f}±{std_score:.4f} "
                                    f"({len(valid_scores)}/{k_folds} folds successful)")
                        else:
                            if verbose:
                                print(f"  Stage 2 Candidate {cand_idx+1}/{n_candidates_s2}: All folds failed")
                    else:
                        if verbose:
                            print(f"  Stage 2 Candidate {cand_idx+1}/{n_candidates_s2}: {status}")
            
            # Find best from stage 2
            mean_cv_scores_s2 = []
            std_cv_scores_s2 = []
            
            for cand_idx in range(n_candidates_s2):
                fold_scores = cv_scores_s2[cand_idx, :]
                valid_scores = fold_scores[np.isfinite(fold_scores)]
                
                if len(valid_scores) > 0:
                    mean_cv_scores_s2.append(np.mean(valid_scores))
                    std_cv_scores_s2.append(np.std(valid_scores))
                else:
                    mean_cv_scores_s2.append(np.inf)
                    std_cv_scores_s2.append(np.inf)
            
            mean_cv_scores_s2 = np.array(mean_cv_scores_s2)
            std_cv_scores_s2 = np.array(std_cv_scores_s2)
            
            # Compare stage 2 best with stage 1 best
            if not np.all(np.isinf(mean_cv_scores_s2)):
                best_idx_s2 = np.argmin(mean_cv_scores_s2)
                best_hyperparams_s2 = stage2_hyperparam_candidates[best_idx_s2]
                best_score_s2 = mean_cv_scores_s2[best_idx_s2]
                best_score_std_s2 = std_cv_scores_s2[best_idx_s2]
                
                if verbose:
                    print(f"\nStage 2 best hyperparameters (candidate {best_idx_s2+1}): {best_hyperparams_s2}")
                    print(f"Stage 2 best CV {scoring}: {best_score_s2:.4f} ± {best_score_std_s2:.4f}")
                
                # Use stage 2 results if better
                if best_score_s2 < best_score:  # Assuming lower is better for most metrics
                    best_hyperparams = best_hyperparams_s2
                    best_score = best_score_s2
                    best_score_std = best_score_std_s2
                    best_idx = best_idx_s2  # Note: this is index within stage 2 candidates
                    if verbose:
                        print(f"✓ Stage 2 improved results!")
                else:
                    if verbose:
                        print(f"✓ Stage 1 results remain best.")
            else:
                if verbose:
                    print("Stage 2 failed - using Stage 1 results")
        
        # Configure final GP with best hyperparameters
        try:
            gp.set_parameter_vector(best_hyperparams)
            gp.compute(_theta)
        except Exception as e:
            raise RuntimeError(f"Failed to set best hyperparameters: {str(e)}")
        
        # Compile detailed results
        cv_results = {
            'best_score': best_score,
            'best_score_std': best_score_std,
            'best_hyperparams': best_hyperparams,
            'best_candidate_idx': best_idx,
            'all_scores': mean_cv_scores,
            'all_scores_std': std_cv_scores,
            'cv_scores_matrix': cv_scores,
            'scoring_method': scoring,
            'n_folds': k_folds,
            'n_candidates': n_candidates,
            'two_stage': two_stage
        }
        
        # Add stage-specific results if two-stage was used
        if two_stage:
            cv_results['stage1_results'] = stage1_results
            if 'mean_cv_scores_s2' in locals():
                cv_results['stage2_results'] = {
                    'all_scores': mean_cv_scores_s2,
                    'all_scores_std': std_cv_scores_s2,
                    'cv_scores_matrix': cv_scores_s2,
                    'candidates': stage2_hyperparam_candidates
                }
        
        return gp, best_hyperparams, cv_results


def _generate_stage2_candidates(best_params, n_candidates, width_factor, random_state):
    """
    Generate stage 2 candidates around best parameters from stage 1.
    
    :param best_params: Best hyperparameters from stage 1
    :param n_candidates: Number of candidates to generate
    :param width_factor: Width of search around best parameters
    :param random_state: Random seed
    :returns: Array of stage 2 hyperparameter candidates
    """
    np.random.seed(random_state + 1000)  # Different seed for stage 2
    
    best_params = np.asarray(best_params)
    n_params = len(best_params)
    
    # Generate candidates using normal distribution around best parameters
    candidates = []
    
    # Always include the best parameters as first candidate
    candidates.append(best_params.copy())
    
    # Generate remaining candidates
    for i in range(n_candidates - 1):
        # Add random perturbations to each parameter
        perturbations = np.random.normal(0, width_factor, n_params)
        candidate = best_params + perturbations
        candidates.append(candidate)
    
    return np.array(candidates)
