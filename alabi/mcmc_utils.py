import numpy as np


# ================================
# probability functions
# ================================

def lnprob(theta, lnlike, lnprior):

    return lnprior(theta) + lnlike(theta)

# ================================
# emcee utils
# ================================

def estimateBurnin(sampler, estBurnin=True, thinChains=True, verbose=False):
    """
    Estimate the integrated autocorrelation length on the MCMC chain associated
    with an emcee sampler object. With the integrated autocorrelation length,
    we can then estimate the burn-in length for the MCMC chain. This procedure
    follows the example outlined here:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        emcee MCMC sampler object/backend handler, given a complete chain
    estBurnin : bool, optional
        Estimate burn-in time using integrated autocorrelation time
        heuristic.  Defaults to True. In general, we recommend users
        inspect the chains and calculate the burnin after the fact to ensure
        convergence, but this function works pretty well.
    thinChains : bool, optional
        Whether or not to thin chains.  Useful if running long chains.
        Defaults to True.  If true, estimates a thin cadence
        via int(0.5*np.min(tau)) where tau is the intergrated autocorrelation
        time.
    verbose : bool, optional
        Output all the diagnostics? Defaults to False.

    Returns
    -------
    iburn : int
        burn-in index estimate.  If estBurnin == False, returns 0.
    ithin : int
        thin cadence estimate.  If thinChains == False, returns 1.
    """

    # Set tol = 0 so it always returns an answer
    tau = sampler.get_autocorr_time(tol=0)

    # Catch NaNs
    if np.any(~np.isfinite(tau)):
        # Try removing NaNs
        tau = tau[np.isfinite(np.array(tau))]
        if len(tau) < 1:
            if verbose:
                print("Failed to compute integrated autocorrelation length, tau.")
                print("Setting tau = 1")
            tau = 1

    # Estimate burn-in?
    if estBurnin:
        iburn = int(2.0*np.max(tau))
    else:
        iburn = 0

    # Thin chains?
    if thinChains:
        ithin = np.max((int(0.5*np.min(tau)), 1))
    else:
        ithin = 1

    if verbose:
        print("burn-in estimate: %d" % iburn)
        print("thin estimate: %d\n" % ithin)

    return iburn, ithin