import numpy as np
from scipy.stats import qmc, gaussian_kde
from scipy import integrate
import os
from .cache_utils import load_model_cache


__all__ = ["kl_divergence_gaussian",
           "js_divergence_gaussian",
           "kl_divergence_integral",
           "kl_divergence_kde",
           "compute_kl_single_trial_joblib",
           "compute_kl_full_parallel"]

def kl_divergence_gaussian(mu1, cov1, mu2, cov2, reg=1e-6):
    """
    Compute the Kullback-Leibler divergence between two Gaussian distributions.

    :param mu1: (*array-like*) Mean of the first Gaussian.
    :param cov1: (*array-like*) Covariance of the first Gaussian.
    :param mu2: (*array-like*) Mean of the second Gaussian.
    :param cov2: (*array-like*) Covariance of the second Gaussian.

    :returns: (*float*) Kullback-Leibler divergence D_KL(N(mu1, cov1) || N(mu2, cov2)).
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)

    # Regularize the covariance matrices
    cov1 += reg * np.eye(cov1.shape[0])
    cov2 += reg * np.eye(cov2.shape[0])

    # Compute the determinant of the covariance matrices
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)

    # Compute the inverse of the second covariance matrix
    inv_cov2 = np.linalg.inv(cov2)

    # Compute the Kullback-Leibler divergence
    kl_div = 0.5 * (np.log(det2 / det1) - len(mu1) + np.trace(inv_cov2 @ cov1) + (mu2 - mu1).T @ inv_cov2 @ (mu2 - mu1))

    return kl_div


def js_divergence_gaussian(mu1, cov1, mu2, cov2):
    """
    Compute the Jensen-Shannon divergence between two Gaussian distributions.

    :param mu1: (*array-like*) Mean of the first Gaussian.
    :param cov1: (*array-like*) Covariance of the first Gaussian.
    :param mu2: (*array-like*) Mean of the second Gaussian.
    :param cov2: (*array-like*) Covariance of the second Gaussian.

    :returns: (*float*) Jensen-Shannon divergence JSD(N(mu1, cov1) || N(mu2, cov2)).
    """
    mu_avg = (mu1 + mu2) / 2
    cov_avg = (cov1 + cov2) / 2

    kl1 = kl_divergence_gaussian(mu1, cov1, mu_avg, cov_avg)
    kl2 = kl_divergence_gaussian(mu2, cov2, mu_avg, cov_avg)

    return (kl1 + kl2) / 2
    
    
def kl_divergence_integral(log_p, log_q, bounds, method='qmc', 
                           n_samples=int(2**14), epsilon=1e-12,
                           n_jobs=1):
    """
    Numerically compute KL divergence: KL(P||Q) = ∫ p(x) * log(p(x)/q(x)) dx
    
    This function estimates the Kullback-Leibler divergence between two probability
    distributions using numerical integration. The KL divergence measures how one
    probability distribution P diverges from a second distribution Q.
    
    :param log_p: Log probability density function P(x). Should return log(p(x)) for input x.
    :type log_p: *callable*
    :param log_q: Log probability density function Q(x). Should return log(q(x)) for input x.
    :type log_q: *callable*
    :param bounds: Integration bounds. For 1D: [a, b]. For nD: [[a1, b1], [a2, b2], ...]
    :type bounds: *array-like*
    :param method: Integration method:
        
        - 'quad': scipy.integrate (exact for 1D, nquad for multi-D)
        - 'mc': Monte Carlo sampling
        - 'qmc': Quasi-Monte Carlo (Sobol sequence) 
        
        Default is 'qmc'.
    :type method: *str, optional*
    :param n_samples: Number of samples for MC/QMC methods. Default is 2^14 = 16384.
    :type n_samples: *int, optional*
    :param epsilon: Small value to avoid log(0) issues. Default is 1e-12.
    :type epsilon: *float, optional*
    :param n_jobs: Number of parallel jobs (reserved for future use). Default is 1.
    :type n_jobs: *int, optional*
    
    :returns:
        - **kl_div** (*float*) -- KL divergence estimate D_KL(P||Q).
        - **error** (*float*) -- Error estimate (for 'quad' method) or standard error estimate (for MC/QMC methods).
    :rtype: *tuple*
    
    .. note::
        
        The KL divergence is defined as:
        
        .. math::
            
            D_{KL}(P||Q) = \\int p(x) \\log\\left(\\frac{p(x)}{q(x)}\\right) dx
        
        This function handles numerical stability by capping very large values and
        avoiding log(0) issues through the epsilon parameter.
    
    **Examples**
    
    1D integration with quadrature:
    
    .. code-block:: python
        
        >>> import numpy as np
        >>> from scipy.stats import norm
        >>> 
        >>> # Define log probability functions for two normal distributions
        >>> log_p = lambda x: norm.logpdf(x, loc=0, scale=1)
        >>> log_q = lambda x: norm.logpdf(x, loc=1, scale=1.5)
        >>> 
        >>> bounds = np.array([-5, 5])
        >>> kl_div, error = kl_divergence_integral(log_p, log_q, bounds, method='quad')
    
    Multi-dimensional with Quasi-Monte Carlo:
    
    .. code-block:: python
        
        >>> # 2D case
        >>> bounds = np.array([[-3, 3], [-3, 3]])
        >>> kl_div, error = kl_divergence_integral(log_p_2d, log_q_2d, bounds, 
        ...                                        method='qmc', n_samples=10000)
    """
        
    def integrand(x):
        if np.isscalar(x):
            x = np.array([x])

        p_val = np.exp(log_p(x))
        q_val = np.exp(log_q(x))

        # Avoid numerical issues with very small probabilities
        p_val = np.maximum(p_val, epsilon)
        q_val = np.maximum(q_val, epsilon)
        
        return p_val * np.log(p_val / q_val)
    
    # 1D integration using scipy.integrate.quad
    if method == 'quad' and bounds.ndim == 1:
        result, error = integrate.quad(integrand, bounds[0], bounds[1])
        return result, error
    
    # Multi-dimensional integration using scipy
    elif method == 'quad' and bounds.ndim == 2:
        def integrand_nd(*args):
            return integrand(np.array(args))
        
        result, error = integrate.nquad(integrand_nd, bounds)
        return result, error
    
    # Monte Carlo integration
    elif (method == 'mc') or (method == 'qmc'):
        
        if bounds.ndim == 1:
            bounds = bounds.reshape(1, -1)
        ndim = bounds.shape[0]
        
        if method == 'mc':
            # Generate random samples uniformly within bounds
            samples = np.random.uniform(
                low=bounds[:, 0], 
                high=bounds[:, 1], 
                size=(n_samples, ndim)
            )
        elif method == 'qmc':
            # Generate Sobol sequence for quasi-Monte Carlo
            sampler = qmc.Sobol(d=ndim, scramble=True)
            unit_samples = sampler.random(n_samples)
            
            # Transform to actual bounds
            samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
        
        # Calculate volume
        volume = np.prod(bounds[:, 1] - bounds[:, 0])
        
        # Evaluate integrand at sample points
        integrand_vals = np.array([integrand(sample) for sample in samples])
        
        integrand_vals[integrand_vals > 1e10] = np.nan  # Cap large values to avoid overflow
        integrand_vals[integrand_vals < 0] = np.nan

        # QMC estimate
        kl_estimate = volume * np.nanmean(integrand_vals) 
        
        # Rough error estimate
        error_estimate = volume * np.nanstd(integrand_vals) / np.sqrt(n_samples)

        return kl_estimate, error_estimate

    else:
        raise ValueError("Invalid method. Choose 'quad', 'mc', or 'qmc'")


def kl_divergence_kde(samples_p, samples_q, bandwidth=None, epsilon=1e-12, n_eval=1000):
    """
    Compute KL divergence between two sets of samples from distributions P and Q
    using kernel density estimation (KDE).
    
    This function estimates the KL divergence by first fitting Gaussian kernel density
    estimators to both sample sets, then computing the expectation E_P[log(P/Q)] over
    the samples from distribution P. This approach is useful when you have samples
    from two distributions but don't have explicit probability density functions.
    
    :param samples_p: Samples from distribution P. For 1D data, can be 1D array.
        For multi-dimensional data, should be shape (n_samples, n_dimensions).
    :type samples_p: *array-like, shape (n_samples_p, n_dims)*
    :param samples_q: Samples from distribution Q. For 1D data, can be 1D array.
        For multi-dimensional data, should be shape (n_samples, n_dimensions).
    :type samples_q: *array-like, shape (n_samples_q, n_dims)*
    :param bandwidth: Bandwidth for the KDE. If None, it will be estimated automatically
        using Scott's rule or Silverman's rule. Default is None.
    :type bandwidth: *float, optional*
    
    :returns: KL divergence D_KL(P||Q) = E_P[log(P/Q)]
    :rtype: *float*
    
    .. note::
        
        The KL divergence is estimated as:
        
        .. math::
            
            \\hat{D}_{KL}(P||Q) = \\frac{1}{N_P} \\sum_{i=1}^{N_P} \\log\\left(\\frac{\\hat{p}(x_i^P)}{\\hat{q}(x_i^P)}\\right)
        
        where :math:`\\hat{p}` and :math:`\\hat{q}` are the KDE estimates of the densities,
        and :math:`x_i^P` are samples from distribution P.
        
        This method works well when:
        
        - You have sufficient samples from both distributions
        - The distributions are reasonably smooth
        - The dimensionality is not too high (curse of dimensionality affects KDE)
    
    **Examples**
    
    Compare two 1D normal distributions:
    
    .. code-block:: python
        
        >>> import numpy as np
        >>> 
        >>> # Generate samples from two normal distributions
        >>> samples_p = np.random.normal(0, 1, 1000)  # N(0,1)
        >>> samples_q = np.random.normal(1, 1.5, 1000)  # N(1,1.5²)
        >>> 
        >>> kl_div = kl_divergence_kde(samples_p, samples_q)
        >>> print(f"KL divergence: {kl_div:.4f}")
    
    Multi-dimensional case:
    
    .. code-block:: python
        
        >>> # 2D multivariate normal distributions
        >>> mean_p, cov_p = [0, 0], [[1, 0.5], [0.5, 1]]
        >>> mean_q, cov_q = [1, 1], [[1.5, 0], [0, 1.5]]
        >>> 
        >>> samples_p = np.random.multivariate_normal(mean_p, cov_p, 1000)
        >>> samples_q = np.random.multivariate_normal(mean_q, cov_q, 1000)
        >>> 
        >>> kl_div = kl_divergence_kde(samples_p, samples_q, bandwidth=0.3)
    """
    
    samples_p = np.asarray(samples_p)
    samples_q = np.asarray(samples_q)
    
    if samples_p.ndim == 1:
        samples_p = samples_p.reshape(-1, 1)
    if samples_q.ndim == 1:
        samples_q = samples_q.reshape(-1, 1)

    if samples_p.shape[1] != samples_q.shape[1]:
        raise ValueError("Samples must have same dimensionality")

    # Fit KDE
    if bandwidth is not None:
        kde_p = gaussian_kde(samples_p.T, bw_method="scott")
        kde_q = gaussian_kde(samples_q.T, bw_method="scott")
    else:
        kde_p = gaussian_kde(samples_p.T)
        kde_q = gaussian_kde(samples_q.T)
    
    # Method 1: Use a separate evaluation set (recommended)
    # Create evaluation points by sampling from the combined range
    n_dim = samples_p.shape[1]
    all_samples = np.vstack([samples_p, samples_q])
    
    # Generate evaluation points in the support region
    min_vals = np.min(all_samples, axis=0)
    max_vals = np.max(all_samples, axis=0)
    
    # Random evaluation points in the support
    eval_samples = np.random.uniform(
        min_vals, max_vals, size=(n_eval, n_dim)
    )
    eval_points = eval_samples.T
    
    # Evaluate KDEs at independent points
    pdf_p = kde_p.pdf(eval_points)
    pdf_q = kde_q.pdf(eval_points)
    
    # Both pdf_p and pdf_q are already normalized by gaussian_kde
    # Ensure positive probabilities to avoid log(0)
    pdf_p = np.maximum(pdf_p, epsilon)
    pdf_q = np.maximum(pdf_q, epsilon)
    
    # Compute KL divergence: E_P[log(P/Q)]
    # Since we're evaluating at random points, we approximate the expectation
    # by weighting by the probability density pdf_p
    log_ratio = np.log(pdf_p / pdf_q)
    valid_mask = np.isfinite(log_ratio)
    
    if np.sum(valid_mask) == 0:
        return np.nan
    
    # The weights should be pdf_p normalized by the sum of pdf_p values
    # to ensure we're approximating the expectation under P
    weights = pdf_p[valid_mask] / np.sum(pdf_p[valid_mask])
    kl_div = np.sum(weights * log_ratio[valid_mask])

    return np.abs(kl_div) 


def compute_kl_single_trial_joblib(trial, ii, base_dir, example, kernel):
    """Compute KL divergence for a single trial and iteration (joblib version)"""
    
    file_p = f"{base_dir}/{example}/{kernel}/{trial}/dynesty_samples_final_surrogate_iter_{ii}.npz"
    file_q = f"{base_dir}/{example}/{kernel}/dynesty_samples_final_true.npz"
    
    if os.path.exists(file_p) is False:
        sm = load_model_cache(f"{base_dir}/{example}/{kernel}/{trial}/")
        print(f"Loaded model from cache for trial {trial}, iteration {ii}")
        sm.run_dynesty(like_fn=sm.surrogate_log_likelihood)
        
    if os.path.exists(file_q) is False:
        sm = load_model_cache(f"{base_dir}/{example}/{kernel}/")
        print(f"Loaded true model from cache for trial {trial}, iteration {ii}")
        sm.run_dynesty(like_fn=sm.lnlike_fn)
        
    try:
        samples_p = np.load(f"{base_dir}/{example}/{kernel}/{trial}/dynesty_samples_final_surrogate_iter_{ii}.npz")["samples"]
        samples_q = np.load(f"{base_dir}/{example}/{kernel}/dynesty_samples_final_true.npz")["samples"]
        
        return kl_divergence_kde(samples_p, samples_q)
    except Exception as e:
        print(f"Error processing trial {trial}, iteration {ii}: {e}")
        return np.nan


def compute_kl_full_parallel(base_dir, example, kernel, trials=np.arange(0, 30), 
                           iterations=np.arange(10, 250, 10), n_jobs=16):
    """Compute all KL divergences for a configuration in parallel"""
    
    from joblib import Parallel, delayed
    
    # Create all (trial, iteration) combinations
    all_tasks = [(trial, ii, base_dir, example, kernel) 
                 for trial in trials 
                 for ii in iterations]
    
    print(f"Processing {len(all_tasks)} tasks for {base_dir}/{example}/{kernel}")
    
    # Parallel execution over all trial-iteration combinations
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(compute_kl_single_trial_joblib)(trial, ii, base_dir, example, kernel)
        for trial, ii, base_dir, example, kernel in all_tasks
    )
    
    # Reorganize results by iteration
    kl_by_iteration = {}
    for idx, (trial, ii, _, _, _) in enumerate(all_tasks):
        if ii not in kl_by_iteration:
            kl_by_iteration[ii] = []
        kl_by_iteration[ii].append(results[idx])
    
    # Compute mean and std for each iteration
    iteration_results = []
    for ii in iterations:
        valid_kl = [kl for kl in kl_by_iteration[ii] if not np.isnan(kl)]
        if len(valid_kl) > 0:
            stat_results = np.array([np.mean(valid_kl), np.std(valid_kl), np.percentile(valid_kl, 25), np.median(valid_kl), np.percentile(valid_kl, 75)])
            iteration_results.append(stat_results)
        else:
            iteration_results.append([np.nan, np.nan, np.nan, np.nan, np.nan])

    return np.array(iteration_results)