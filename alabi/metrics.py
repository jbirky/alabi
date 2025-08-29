import numpy as np
from scipy.stats import qmc, gaussian_kde
from scipy import integrate


__all__ = ["kl_divergence_gaussian",
           "js_divergence_gaussian",
           "kl_divergence_kde",
           "kl_divergence_integral",]

def kl_divergence_gaussian(mu1, cov1, mu2, cov2):
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
    
    Parameters:
    -----------
    p : callable
        Probability density function P(x)
    q : callable
        Probability density function Q(x)
    bounds : array-like
        Integration bounds. For 1D: [a, b]. For nD: [[a1, b1], [a2, b2], ...]
    method : str, default='quad'
        Integration method: 'quad' (scipy), 'mc' (Monte Carlo), 'qmc' (Quasi-Monte Carlo)
    n_samples : int, default=10000
        Number of samples for MC/QMC methods
    epsilon : float, default=1e-12
        Small value to avoid log(0) issues
        
    Returns:
    --------
    kl_div : float
        KL divergence estimate
    error : float (optional)
        Error estimate for quad method
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


def kl_divergence_kde(samples_p, samples_q, bandwidth=None):
    """
    Compute KL divergence between two sets of samples from distributions P and Q
    using kernel density estimation (KDE).
    
    Parameters:
    -----------
    samples_p : array-like, shape (n_samples_p, n_dims)
        Samples from distribution P
    samples_q : array-like, shape (n_samples_q, n_dims)
        Samples from distribution Q
    bandwidth : float, optional
        Bandwidth for the KDE. If None, it will be estimated automatically.
        Default is None.
    
    Returns:
    --------
    kl_div : float
        KL divergence D_KL(P||Q) = E_P[log(P/Q)]
    """
    
    samples_p = np.asarray(samples_p)
    samples_q = np.asarray(samples_q)
    
    if samples_p.ndim == 1:
        samples_p = samples_p.reshape(-1, 1)
    if samples_q.ndim == 1:
        samples_q = samples_q.reshape(-1, 1)
    
    # Fit KDEs to both distributions
    kde_p = gaussian_kde(samples_p.T, bw_method=bandwidth)
    kde_q = gaussian_kde(samples_q.T, bw_method=bandwidth)
    
    # Evaluate densities at P samples
    log_p = kde_p.logpdf(samples_p.T)
    log_q = kde_q.logpdf(samples_p.T)
    
    # KL divergence: E_P[log(P/Q)] = E_P[log(P) - log(Q)]
    kl_div = np.mean(log_p - log_q)
    
    return kl_div
