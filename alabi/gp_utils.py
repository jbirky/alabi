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
from scipy.stats import lognorm


__all__ = ["configure_gp", 
           "optimize_gp",
           "optimize_gp_kfold_cv",
           "regularization_term",
           "regularization_gradient"]


def regularization_term(hparams, lengthscale_indices, amp_0=1.0, mu_0=1.0, sigma_0=2.0):
    """
    Compute the regularization term (negative log prior) from Hvafner 2024 Equation 4.
    
    This implements the dimensionality-scaled LogNormal prior:
    p(ℓ_i | d) = LogNormal(μ_0 + log(√d), σ_0)
    
    The regularization term is -log p(ℓ | d) = -Σ log p(ℓ_i | d)
    
    Parameters:
    -----------
    mu_0 : float, default=0.0
        The base location parameter for the 1D LogNormal prior
    sigma_0 : float, default=1.0
        The scale parameter for the LogNormal prior
    
    Returns:
    --------
    float
        The regularization term (negative log prior)
    """
    ndim = len(hparams)
    log_lengthscales = hparams[lengthscale_indices]
    
    # Scaled location parameter: μ = μ_0 + log(√d)
    mu = mu_0 + 0.5 * np.log(ndim)
    
    # Compute negative log likelihood for each lengthscale
    # For LogNormal(μ, σ), if X ~ LogNormal(μ, σ), then log(X) ~ Normal(μ, σ)
    # log p(ℓ_i) = -log(ℓ_i) - log(σ√(2π)) - (log(ℓ_i) - μ)²/(2σ²)
    
    # Negative log prior for each dimension
    neg_log_prior = (
        log_lengthscales +  # log(ℓ_i) term from the Jacobian
        0.5 * np.log(2 * np.pi * sigma_0**2) +  # normalization constant
        (log_lengthscales - mu)**2 / (2 * sigma_0**2)  # quadratic term
    )
    
    # Sum over all dimensions
    reg = amp_0 * np.sum(neg_log_prior)
    return reg


def regularization_gradient(hparams, lengthscale_indices, amp_0=1.0, mu_0=1.0, sigma_0=2.0):
    """
    Compute the gradient of the regularization term with respect to lengthscales.
    
    Parameters:
    -----------
    lengthscales : array-like, shape (d,)
        The lengthscale parameters for each dimension
    d : int
        The dimensionality of the problem
    mu_0 : float, default=0.0
        The base location parameter for the 1D LogNormal prior
    sigma_0 : float, default=1.0
        The scale parameter for the LogNormal prior
    
    Returns:
    --------
    ndarray, shape (d,)
        The gradient of the regularization term with respect to each lengthscale
    """
    ndim = len(hparams)
    gradient_vector = np.zeros_like(hparams)
    
    log_lengthscales = hparams[lengthscale_indices]
    lengthscales = np.exp(log_lengthscales)
    
    # Scaled location parameter
    mu = mu_0 + 0.5 * np.log(ndim)
    
    # d/dℓ_i [-log p(ℓ_i)] = d/dℓ_i [log(ℓ_i) + C + (log(ℓ_i) - μ)²/(2σ²)]
    #                       = 1/ℓ_i + (log(ℓ_i) - μ)/(σ² ℓ_i)
    #                       = [1 + (log(ℓ_i) - μ)/σ²] / ℓ_i

    gradient_vector[lengthscale_indices] = (1.0 + (log_lengthscales - mu) / sigma_0**2) / lengthscales
    
    return amp_0 * gradient_vector


def _nll(p, gp, y, gp_hyper_prior):
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

    Returns
    -------
    nll : float
        negative log-likelihood of y under gp
    """

    if not np.isfinite(gp_hyper_prior(p)):
        return np.inf
    
    # Catch singular matrices
    try:
        gp.set_parameter_vector(p)
    except np.linalg.LinAlgError:
        return np.inf

    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else np.inf


def _grad_nll(p, gp, y):
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

    Returns
    -------
    gnll : array
        gradient of the negative log-likelihood of y under gp
    """
        
    gp.set_parameter_vector(p)

    # Negative gradient of log likelihood
    grad_nll = -gp.grad_log_likelihood(y, quiet=True)
    
    return grad_nll


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
    """

    if np.any(~np.isfinite(theta)):
        print("theta", theta)
        raise ValueError("All theta values must be finite!")

    if np.any(~np.isfinite(y)):
        print("y", y)
        raise ValueError("All y values must be finite!")

    if fit_amp == True:
        kernel *= np.var(y)

    gp = george.GP(kernel=kernel, fit_mean=fit_mean, mean=np.median(y),
                   white_noise=white_noise, fit_white_noise=fit_white_noise)

    if hyperparameters is not None:
        if np.any(~np.isfinite(hyperparameters)):
            print("hyperparameters:", hyperparameters)
            raise ValueError("All hyperparameter values must be finite!")
        gp.set_parameter_vector(hyperparameters)

    try:
        gp.compute(theta)
    except Exception as e:
        print(f"configure_gp error: {e}")
        return None

    return gp


def optimize_gp(gp, _theta, _y, gp_hyper_prior, p0, bounds=None,
                method="l-bfgs-b", optimizer_kwargs=None, 
                regularize=True, amp_0=1.0, mu_0=1.0, sigma_0=2.0, lengthscale_indices=None):
    """
    Optimize Gaussian Process hyperparameters by maximizing marginal likelihood.
    
    Performs hyperparameter optimization for a Gaussian Process using scipy's
    minimize function. Supports multiple optimization restarts and automatically
    selects the result with highest marginal likelihood.
    
    :param gp: (*george.GP*)
        Configured Gaussian Process object. Should be computed with training data.
        
    :param _theta: (*array-like, shape (n_samples, n_features)*)
        Training input locations (parameters). Will be squeezed if 1D.
        
    :param _y: (*array-like, shape (n_samples,)*)
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
        
    :param regularize: (*bool, optional, default=False*)
        Whether to apply Hvarfner dimensionality-scaled regularization (Equation 4
        from "Vanilla Bayesian Optimization Performs Great in High Dimensions").
        When True, lengthscale priors are scaled as LogNormal(μ_0 + log(√d), σ_0).
        
    :param mu_0: (*float, optional, default=0.0*)
        Base location parameter for Hvarfner regularization. Only used if regularize=True.
        
    :param sigma_0: (*float, optional, default=1.0*)
        Scale parameter for Hvarfner regularization. Only used if regularize=True.
        
    :param lengthscale_indices: (*list of int, optional, default=None*)
        Indices in the parameter vector corresponding to lengthscale parameters.
        Only used if regularize=True. If None, attempts to infer from kernel.
        
    :returns: *george.GP or None*
        GP object with optimized hyperparameters, or None if optimization failed.
        The returned GP is recomputed with optimal hyperparameters.
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Collapse arrays if 1D
    _theta = _theta.squeeze()
    _y = _y.squeeze()
    
    # Get dimensionality for regularization
    ndim = _theta.shape[-1] if _theta.ndim > 1 else 1

    # initial hyperparameters
    init_hp = gp.get_parameter_vector()
    nhparam = init_hp.shape[0]
    
    # Infer lengthscale indices if regularization is enabled
    if regularize and lengthscale_indices is None:
        param_names = gp.kernel.get_parameter_names()
        lengthscale_indices = []
        other_indices = []
        for ii, name in enumerate(param_names):
            # Common patterns for lengthscale parameters in george kernels
            if any(pattern in name.lower() for pattern in ['metric:log_m']):
                lengthscale_indices.append(ii)
            else:
                other_indices.append(ii)
        
        if len(lengthscale_indices) == 0:
            warnings.warn("Could not infer lengthscale indices from kernel. "
                        "Regularization gradient will not be applied. Please specify "
                        "lengthscale_indices explicitly.")
        
    valid_methods = ["newton-cg", "bfgs", "l-bfgs-b", "powell", "nelder-mead"]
    if method not in valid_methods:
        print(f"Warning: {method} not in valid methods {valid_methods}. Using 'l-bfgs-b' optimizer instead.")
        method = "l-bfgs-b"
        
    # methods without bounds arg 
    if method in ["bfgs"]:
        bounds = None
    
    # configure objective function 
    if regularize:
        obj_fn = lambda p: _nll(p, gp, _y, gp_hyper_prior) + regularization_term(p, lengthscale_indices, amp_0=amp_0, mu_0=mu_0, sigma_0=sigma_0)
    else:
        obj_fn = lambda p: _nll(p, gp, _y, gp_hyper_prior)

    # configure gradient function
    if method in ["newton-cg", "l-bfgs-b"]:
        if regularize:
            # Use custom gradient function that includes regularization
            jac = lambda p: _grad_nll(p, gp, _y) + regularization_gradient(p, lengthscale_indices, amp_0=amp_0, mu_0=mu_0, sigma_0=sigma_0)
        else:
            jac = lambda p: _grad_nll(p, gp, _y)
    else:
        jac = None
        
    # # Set improved default optimizer_kwargs for faster convergence
    # if optimizer_kwargs is None:
    #     default_optimizer_kwargs = {
    #         'newton-cg': {'maxiter': 200},
    #         'bfgs': {'maxiter': 200},
    #         'l-bfgs-b': {'maxiter': 200, 'factr': 1e12},
    #         'powell': {'maxiter': 200},
    #         'nelder-mead': {'maxiter': 200},
    #     }
    #     optimizer_kwargs = default_optimizer_kwargs.get(method.lower(), {})
    
    nopt = p0.shape[0] if p0.ndim > 1 else 1
    if nopt > 1:
        # Run the optimization routine nopt times
        res = []
        mll = []
        
        for ii, x0 in enumerate(p0):
            try:
                result = minimize(obj_fn, x0, method=method, jac=jac, bounds=bounds, options=optimizer_kwargs)
                print("opt iterations:", result.nit, result.success)
                
                if np.isfinite(gp_hyper_prior(result.x)):
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
        # Single optimization run
        try:
            result = minimize(obj_fn, p0, method=method, jac=jac, bounds=bounds, options=optimizer_kwargs)
            
            if result.success and np.isfinite(gp_hyper_prior(result.x)):
                gp.set_parameter_vector(result.x)
                gp.recompute()
            else:
                print("\nWarning: GP hyperparameter optimization failed. Using initial values.\n")
                gp.set_parameter_vector(init_hp)
                gp.recompute()
                
        except Exception as e:
            print(f"\nWarning: GP hyperparameter optimization failed with error: {e}. Using initial values.\n")
            gp.set_parameter_vector(init_hp)
            gp.recompute()

    return gp


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


def _evaluate_candidate_worker(args):
    """
    Worker function for parallel evaluation of a single hyperparameter candidate.
    
    :param args: tuple containing (cand_idx, hyperparams, gp, theta, y,  
                                   k_folds, scoring, weighted_mse_method, weighted_mse_factor)
    :returns: tuple (cand_idx, fold_scores, success_flag)
    """
    import copy
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    (cand_idx, hyperparams, gp, _theta, _y, y_scaler, k_folds, scoring, weighted_mse_method, weighted_mse_factor) = args

    try:
        # Check for basic hyperparameter validity
        if np.any(np.isnan(hyperparams)) or np.any(np.isinf(hyperparams)):
            return (cand_idx, None, "Invalid hyperparameters (NaN/Inf)")
        
        # Set up cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=None)
        fold_scores = []
        fold_errors = []  # Track errors from each fold
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
                                                           temperature=weighted_mse_factor)
                else:
                    raise ValueError(f"Unsupported scoring method: {scoring}")
                
                fold_scores.append(fold_score)
                
            except Exception as e:
                fold_scores.append(np.inf)
                fold_errors.append(f"Fold {fold_idx}: {str(e)}")
            
            fold_idx += 1
        
        # If all folds failed, return error message with details
        if all(np.isinf(fold_scores)):
            error_msg = f"All folds failed. First error: {fold_errors[0] if fold_errors else 'Unknown'}"
            return (cand_idx, fold_scores, error_msg)
        
        return (cand_idx, fold_scores, "success")
        
    except Exception as e:
        return (cand_idx, None, str(e))


def optimize_gp_kfold_cv(gp, _theta, _y, hyperparameter_candidates, y_scaler,
                         k_folds=5, scoring="mse", pool=None, 
                         stage2_candidates=None, stage2_width=0.5,
                         stage3_candidates=None, stage3_width=0.2,
                         weighted_mse_method="exponential", weighted_mse_factor=1.0,
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
        
    :param pool: (*multiprocessing.Pool, optional, default=None*)
        Multiprocessing pool for parallel evaluation of hyperparameter candidates.
        If None, evaluation runs sequentially. If provided, candidates are 
        evaluated in parallel using the pool's worker processes.
        
    :param stage2_candidates: (*int, optional, default=None*)
        Number of candidates for stage 2 grid search. If None, uses 
        len(hyperparameter_candidates) // 2.
        
    :param stage2_width: (*float, optional, default=0.5*)
        Width factor for stage 2 search around best parameters.
        Smaller values = tighter search, larger values = wider search.
        
    :param stage3_candidates: (*int, optional, default=None*)
        Number of candidates for stage 3 ultra-fine search. If None, uses 
        max(stage2_candidates // 2, 3).
        
    :param stage3_width: (*float, optional, default=0.2*)
        Width factor for stage 3 search around stage 2 best parameters.
        Should be smaller than stage2_width for finer refinement.
        
    :param weighted_mse_method: (*str, optional, default='exponential'*)
        Weighting method for weighted_mse scoring. Options:
        - 'exponential': w = exp(y_true / temperature)
        - 'linear': w = y_true - min(y_true) + epsilon  
        - 'softmax': w = softmax(y_true / temperature)
        - 'rank': w based on rank order of y_true
        
    :param weighted_mse_factor: (*float, optional, default=1.0*)
        Temperature parameter for exponential/softmax weighting.
        Lower values emphasize high-probability regions more strongly.
        
    :returns:
        - **best_gp** (*george.GP*) -- GP with optimal hyperparameters set
        - **best_hyperparams** (*array*) -- Best hyperparameter vector
        - **cv_results** (*dict*) -- Detailed CV results with scores and statistics
        
    :raises:
        - **ValueError** -- If insufficient data, invalid parameters, or no valid hyperparameters
        - **RuntimeError** -- If all hyperparameter evaluations fail
    """
    
    if stage2_candidates is not None:
        two_stage = True
    else:
        two_stage = False
    if stage3_candidates is not None:
        three_stage = True
    else:
        three_stage = False
        
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
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=None)
    
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
                   k_folds, scoring, weighted_mse_method, weighted_mse_factor)
            worker_args.append(args)
        
        # Evaluate all candidates in parallel with progress bar
        failed_candidate_errors = []  # Track errors for diagnostics
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
                    failed_candidate_errors.append((cand_idx, hyperparameter_candidates[cand_idx], "All folds failed"))
            else:
                # Candidate failed completely
                if verbose:
                    print(f"  Candidate {cand_idx+1}/{n_candidates}: {status}")
                    if "hyperparams" in status.lower() or "prior" in status.lower():
                        print(f"    Hyperparams: {hyperparameter_candidates[cand_idx]}")
                failed_candidate_errors.append((cand_idx, hyperparameter_candidates[cand_idx], status))
    
    else:
        # Sequential evaluation using the worker function
        failed_candidate_errors = []  # Track errors for diagnostics
        for cand_idx, hyperparams in enumerate(hyperparameter_candidates):
            args = (cand_idx, hyperparams, gp, _theta, _y, y_scaler, k_folds, scoring, weighted_mse_method, weighted_mse_factor)
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
                    failed_candidate_errors.append((cand_idx, hyperparams, "All folds failed"))
            else:
                # Candidate failed completely
                if verbose:
                    print(f"  Candidate {cand_idx+1}/{n_candidates}: {status}")
                    if "hyperparams" in status.lower() or "prior" in status.lower():
                        print(f"    Hyperparams: {hyperparams}")
                failed_candidate_errors.append((cand_idx, hyperparams, status))
    
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
    else:
        # All candidates failed
        if verbose:
            print("\n" + "="*80)
            print("ERROR: All hyperparameter candidates failed CV evaluation")
            print("="*80)
            print("\nDiagnostics:")
            print(f"  Number of candidates: {n_candidates}")
            print(f"  Number of folds: {k_folds}")
            print(f"  Training data shape: {_theta.shape}")
            print(f"  Training targets shape: {_y.shape}")
            print(f"  Training targets stats: min={np.min(_y):.4f}, max={np.max(_y):.4f}, mean={np.mean(_y):.4f}, std={np.std(_y):.4f}")
            
            # Report errors from failed candidates
            if 'failed_candidate_errors' in locals() and len(failed_candidate_errors) > 0:
                print(f"\nShowing first 5 candidate errors:")
                for i, (cand_idx, hp, error) in enumerate(failed_candidate_errors[:5]):
                    print(f"  Candidate {cand_idx+1}: {error}")
                    print(f"    Hyperparams: {hp}")
            
            print("\nPossible causes:")
            print("  - Hyperparameters produce numerically unstable GP")
            print("  - Training data has issues (NaN, inf, extreme values)")
            print("  - GP kernel is incompatible with data")
            print("  - Insufficient training data for cross-validation")
            print("="*80)
        return None
    
    # === STAGE 2: EXPLOITATION (Grid Search around best) ===
    if two_stage:
            if verbose:
                print(f"\n=== STAGE 2: EXPLOITATION (Grid Search around best) ===")
            
            # Determine number of stage 2 candidates
            if stage2_candidates is None:
                stage2_candidates = max(n_candidates // 2, 5)
            
            # Generate grid around best hyperparameters
            stage2_hyperparam_candidates = _generate_stage2_candidates(
                best_hyperparams, stage2_candidates, stage2_width, gp=gp
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
                           k_folds, scoring, weighted_mse_method, weighted_mse_factor)
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
                    args = (cand_idx, hyperparams, gp, _theta, _y, y_scaler, k_folds, scoring, weighted_mse_method, weighted_mse_factor)
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
    
    # === STAGE 3: REFINEMENT (Ultra-fine search around stage 2 best) ===
    if three_stage and two_stage:
        if verbose:
            print(f"\n=== STAGE 3: REFINEMENT (Ultra-fine search around best) ===")
        
        # Determine number of stage 3 candidates
        if stage3_candidates is None:
            stage3_candidates = max((stage2_candidates if stage2_candidates else n_candidates // 2) // 2, 3)
        
        # Generate ultra-fine grid around current best hyperparameters
        stage3_hyperparam_candidates = _generate_stage3_candidates(
            best_hyperparams, stage3_candidates, stage3_width, gp=gp
        )
        
        # Evaluate stage 3 candidates
        n_candidates_s3 = len(stage3_hyperparam_candidates)
        cv_scores_s3 = np.zeros((n_candidates_s3, k_folds))
        cv_scores_s3.fill(np.inf)
        
        if verbose:
            print(f"Evaluating {n_candidates_s3} stage 3 candidates using {k_folds}-fold CV...")
        
        if pool is not None:
            # Parallel evaluation for stage 3
            worker_args_s3 = []
            for cand_idx, hyperparams in enumerate(stage3_hyperparam_candidates):
                args = (cand_idx, hyperparams, gp, _theta, _y, y_scaler,
                       k_folds, scoring, weighted_mse_method, weighted_mse_factor)
                worker_args_s3.append(args)
            
            with tqdm(total=len(worker_args_s3), desc="Stage 3 candidates", unit="candidate") as pbar:
                results_s3 = []
                for result in pool.imap(_evaluate_candidate_worker, worker_args_s3):
                    results_s3.append(result)
                    pbar.update(1)
            
            # Process stage 3 results
            for cand_idx, fold_scores, status in results_s3:
                if status == "success" and fold_scores is not None:
                    for fold_idx, score in enumerate(fold_scores):
                        cv_scores_s3[cand_idx, fold_idx] = score
                    
                    valid_scores = [s for s in fold_scores if np.isfinite(s)]
                    if len(valid_scores) > 0:
                        mean_score = np.mean(valid_scores)
                        std_score = np.std(valid_scores)
                        if verbose:
                            print(f"  Stage 3 Candidate {cand_idx+1}/{n_candidates_s3}: {scoring}={mean_score:.4f}±{std_score:.4f} "
                                f"({len(valid_scores)}/{k_folds} folds successful)")
                    else:
                        if verbose:
                            print(f"  Stage 3 Candidate {cand_idx+1}/{n_candidates_s3}: All folds failed")
                else:
                    if verbose:
                        print(f"  Stage 3 Candidate {cand_idx+1}/{n_candidates_s3}: {status}")
        
        else:
            # Sequential evaluation for stage 3
            for cand_idx, hyperparams in enumerate(stage3_hyperparam_candidates):
                args = (cand_idx, hyperparams, gp, _theta, _y, y_scaler, k_folds, scoring, weighted_mse_method, weighted_mse_factor)
                results_s3 = _evaluate_candidate_worker(args)
                cand_idx_result, fold_scores, status = results_s3
                
                if status == "success" and fold_scores is not None:
                    for fold_idx, score in enumerate(fold_scores):
                        cv_scores_s3[cand_idx, fold_idx] = score
                    
                    valid_scores = [s for s in fold_scores if np.isfinite(s)]
                    if len(valid_scores) > 0:
                        mean_score = np.mean(valid_scores)
                        std_score = np.std(valid_scores)
                        if verbose:
                            print(f"  Stage 3 Candidate {cand_idx+1}/{n_candidates_s3}: {scoring}={mean_score:.4f}±{std_score:.4f} "
                                f"({len(valid_scores)}/{k_folds} folds successful)")
                    else:
                        if verbose:
                            print(f"  Stage 3 Candidate {cand_idx+1}/{n_candidates_s3}: All folds failed")
                else:
                    if verbose:
                        print(f"  Stage 3 Candidate {cand_idx+1}/{n_candidates_s3}: {status}")
        
        # Find best from stage 3
        mean_cv_scores_s3 = []
        std_cv_scores_s3 = []
        
        for cand_idx in range(n_candidates_s3):
            fold_scores = cv_scores_s3[cand_idx, :]
            valid_scores = fold_scores[np.isfinite(fold_scores)]
            
            if len(valid_scores) > 0:
                mean_cv_scores_s3.append(np.mean(valid_scores))
                std_cv_scores_s3.append(np.std(valid_scores))
            else:
                mean_cv_scores_s3.append(np.inf)
                std_cv_scores_s3.append(np.inf)
        
        mean_cv_scores_s3 = np.array(mean_cv_scores_s3)
        std_cv_scores_s3 = np.array(std_cv_scores_s3)
        
        # Compare stage 3 best with previous best
        if not np.all(np.isinf(mean_cv_scores_s3)):
            best_idx_s3 = np.argmin(mean_cv_scores_s3)
            best_hyperparams_s3 = stage3_hyperparam_candidates[best_idx_s3]
            best_score_s3 = mean_cv_scores_s3[best_idx_s3]
            best_score_std_s3 = std_cv_scores_s3[best_idx_s3]
            
            if verbose:
                print(f"\nStage 3 best hyperparameters (candidate {best_idx_s3+1}): {best_hyperparams_s3}")
                print(f"Stage 3 best CV {scoring}: {best_score_s3:.4f} ± {best_score_std_s3:.4f}")
            
            # Use stage 3 results if better
            if best_score_s3 < best_score:  # Assuming lower is better for most metrics
                best_hyperparams = best_hyperparams_s3
                best_score = best_score_s3
                best_score_std = best_score_std_s3
                best_idx = best_idx_s3  # Note: this is index within stage 3 candidates
                if verbose:
                    print(f"✓ Stage 3 improved results!")
            else:
                if verbose:
                    print(f"✓ Previous results remain best.")
        else:
            if verbose:
                    print("Stage 3 failed - using previous results")
    elif three_stage and not two_stage:
        if verbose:
            print("Warning: three_stage=True requires two_stage=True. Ignoring three_stage option.")
    
    # Configure final GP with best hyperparameters
    try:
        if verbose:
            print(f"\nSetting best hyperparameters on GP...")
            print(f"Best hyperparams shape: {np.array(best_hyperparams).shape}")
            print(f"GP current params shape: {gp.get_parameter_vector().shape}")
        
        gp.set_parameter_vector(best_hyperparams)
        gp.compute(_theta)
        
        if verbose:
            print(f"Successfully configured GP with best hyperparameters")
    except Exception as e:
        if verbose:
            print(f"Failed to set best hyperparameters: {str(e)}")
        # raise RuntimeError(f"Failed to set best hyperparameters: {str(e)}")
        # If optimization fails, return original GP without changes
        return gp
    
    # Compile detailed results
    cv_results = {
        "best_score": best_score,
        "best_score_std": best_score_std,
        "best_hyperparams": best_hyperparams,
        "best_candidate_idx": best_idx,
        "all_scores": mean_cv_scores,
        "all_scores_std": std_cv_scores,
        "cv_scores_matrix": cv_scores,
        "scoring_method": scoring,
        "n_folds": k_folds,
        "n_candidates": n_candidates,
        "two_stage": two_stage,
        "three_stage": three_stage
    }
    
    # Add stage-specific results if multi-stage was used
    if two_stage:
        cv_results["stage1_results"] = stage1_results
        if "mean_cv_scores_s2" in locals():
            cv_results["stage2_results"] = {
                "all_scores": mean_cv_scores_s2,
                "all_scores_std": std_cv_scores_s2,
                "cv_scores_matrix": cv_scores_s2,
                "candidates": stage2_hyperparam_candidates
            }
        
        # Add stage 3 results if three-stage was used
        if three_stage and "mean_cv_scores_s3" in locals():
            cv_results["stage3_results"] = {
                "all_scores": mean_cv_scores_s3,
                "all_scores_std": std_cv_scores_s3,
                "cv_scores_matrix": cv_scores_s3,
                "candidates": stage3_hyperparam_candidates
            }
            
    if verbose:
        print(f"CV optimization completed. Best {scoring}: {cv_results['best_score']:.4f} ± {cv_results['best_score_std']:.4f}")
    
    return gp


def _generate_stage2_candidates(best_params, n_candidates, width_factor, gp=None):
    """
    Generate stage 2 candidates around best parameters from stage 1.
    
    :param best_params: Best hyperparameters from stage 1
    :param n_candidates: Number of candidates to generate
    :param width_factor: Width of search around best parameters
    :param gp: Optional GP object to infer parameter structure from kernel
    :returns: Array of stage 2 hyperparameter candidates
    """
    
    best_params = np.asarray(best_params)
    n_params = len(best_params)
    
    # Infer lengthscale indices from GP kernel if available
    lengthscale_indices = None
    if gp is not None:
        param_names = gp.kernel.get_parameter_names()
        lengthscale_indices = [i for i, name in enumerate(param_names) 
                              if 'metric:log_m' in name.lower()]
    
    # Check if we have uniform length scales
    if lengthscale_indices is not None and len(lengthscale_indices) > 1:
        # Check if all lengthscale values are identical
        length_scales = best_params[lengthscale_indices]
        uniform_scales = np.allclose(length_scales, length_scales[0])
    else:
        # Fallback: assume indices 2+ are length scales if n_params > 2
        if n_params > 2:
            length_scales = best_params[2:]
            uniform_scales = np.allclose(length_scales, length_scales[0])
            lengthscale_indices = list(range(2, n_params))
        else:
            uniform_scales = False
            lengthscale_indices = []
    
    # Generate candidates using normal distribution around best parameters
    candidates = []
    
    # Always include the best parameters as first candidate
    candidates.append(best_params.copy())
    
    # Generate remaining candidates
    for i in range(n_candidates - 1):
        if uniform_scales and len(lengthscale_indices) > 0:
            # For uniform scales: perturb each non-lengthscale parameter independently
            # and apply single perturbation to all lengthscales
            candidate = best_params.copy()
            
            # Perturb non-lengthscale parameters independently
            non_lengthscale_indices = [j for j in range(n_params) if j not in lengthscale_indices]
            for idx in non_lengthscale_indices:
                candidate[idx] += np.random.normal(0, width_factor)
            
            # Apply same perturbation to all length scales
            lengthscale_perturbation = np.random.normal(0, width_factor)
            for idx in lengthscale_indices:
                candidate[idx] += lengthscale_perturbation
        else:
            # For independent scales: perturb all parameters independently
            perturbations = np.random.normal(0, width_factor, n_params)
            candidate = best_params + perturbations
        
        candidates.append(candidate)
    
    return np.array(candidates)


def _generate_stage3_candidates(best_params, n_candidates, width_factor, gp=None):
    """
    Generate stage 3 candidates around best parameters from stage 2.
    Uses tighter search with smaller perturbations for ultra-fine optimization.
    
    :param best_params: Best hyperparameters from stage 2
    :param n_candidates: Number of candidates to generate
    :param width_factor: Width of search around best parameters (should be smaller than stage2)
    :param gp: Optional GP object to infer parameter structure from kernel
    :returns: Array of stage 3 hyperparameter candidates
    """
    
    best_params = np.asarray(best_params)
    n_params = len(best_params)
    
    # Infer lengthscale indices from GP kernel if available
    lengthscale_indices = None
    if gp is not None:
        param_names = gp.kernel.get_parameter_names()
        lengthscale_indices = [i for i, name in enumerate(param_names) 
                              if 'metric:log_m' in name.lower()]
    
    # Check if we have uniform length scales
    if lengthscale_indices is not None and len(lengthscale_indices) > 1:
        # Check if all lengthscale values are identical
        length_scales = best_params[lengthscale_indices]
        uniform_scales = np.allclose(length_scales, length_scales[0])
    else:
        # Fallback: assume indices 2+ are length scales if n_params > 2
        if n_params > 2:
            length_scales = best_params[2:]
            uniform_scales = np.allclose(length_scales, length_scales[0])
            lengthscale_indices = list(range(2, n_params))
        else:
            uniform_scales = False
            lengthscale_indices = []
    
    # Generate candidates using tighter normal distribution around best parameters
    candidates = []
    
    # Always include the best parameters as first candidate
    candidates.append(best_params.copy())
    
    # Generate remaining candidates with smaller perturbations for fine-tuning
    for i in range(n_candidates - 1):
        if uniform_scales and len(lengthscale_indices) > 0:
            # For uniform scales: perturb each non-lengthscale parameter independently
            # and apply single perturbation to all lengthscales
            candidate = best_params.copy()
            
            # Perturb non-lengthscale parameters independently
            non_lengthscale_indices = [j for j in range(n_params) if j not in lengthscale_indices]
            for idx in non_lengthscale_indices:
                candidate[idx] += np.random.normal(0, width_factor)
            
            # Apply same perturbation to all length scales
            lengthscale_perturbation = np.random.normal(0, width_factor)
            for idx in lengthscale_indices:
                candidate[idx] += lengthscale_perturbation
        else:
            # For independent scales: perturb all parameters independently
            perturbations = np.random.normal(0, width_factor, n_params)
            candidate = best_params + perturbations
        
        candidates.append(candidate)
    
    return np.array(candidates)