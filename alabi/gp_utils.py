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
           "optimize_gp"]


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
    return amp_0 * np.sum(neg_log_prior)


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
        log_prior = prior_fn(p)
        if not np.isfinite(log_prior):
            return np.inf

    # Catch singular matrices
    try:
        gp.set_parameter_vector(p)
    except np.linalg.LinAlgError:
        return np.inf

    ll = gp.log_likelihood(y, quiet=True)
    
    # Return negative log posterior (NLL - log prior)
    if prior_fn is not None:
        return -ll - log_prior if np.isfinite(ll) else np.inf
    else:
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
    d : int, optional
        Dimensionality for Hvarfner regularization gradient
    lengthscale_indices : list of int, optional
        Indices of lengthscale parameters for regularization gradient
    mu_0 : float, default=0.0
        Base location parameter for regularization
    sigma_0 : float, default=1.0
        Scale parameter for regularization

    Returns
    -------
    gnll : array
        gradient of the negative log-likelihood of y under gp
    """

    # Apply priors on GP hyperparameters
    if prior_fn is not None:
        log_prior = prior_fn(p)
        if not np.isfinite(log_prior):
            return np.full_like(p, 0)

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

    if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(y)):
        print("theta, y:", theta, y)
        raise ValueError("All theta and y values must be finite!")

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
        breakpoint()
        return None

    return gp


def optimize_gp(gp, _theta, _y, gp_hyper_prior, p0, bounds=None,
                method="l-bfgs-b", optimizer_kwargs=None, max_iter=50,
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
        
    valid_methods = ["newton-cg", "bfgs", "l-bfgs-b", "powell"]
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
            jac = lambda p: _grad_nll(p, gp, _y, gp_hyper_prior) + regularization_gradient(p, lengthscale_indices, amp_0=amp_0, mu_0=mu_0, sigma_0=sigma_0)
        else:
            jac = lambda p: _grad_nll(p, gp, _y, gp_hyper_prior)
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
                result = minimize(obj_fn, x0, method=method, jac=jac, bounds=bounds, options=optimizer_kwargs)
                
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