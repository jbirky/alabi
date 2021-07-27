"""
:py:mod:`mcmc_utils.py` 
-------------------------------------
"""

import numpy as np

__all__ = ["estimateBurnin"]


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

    :param sampler: (*emcee.EnsembleSampler, optional*)
        emcee MCMC sampler object/backend handler, given a complete chain
    :param estBurnin: (*bool, optional*)
        Estimate burn-in time using integrated autocorrelation time
        heuristic.  Defaults to True. In general, we recommend users
        inspect the chains and calculate the burnin after the fact to ensure
        convergence, but this function works pretty well.
    :param thinChains: (*bool, optional*)
        Whether or not to thin chains.  Useful if running long chains.
        Defaults to True.  If true, estimates a thin cadence
        via int(0.5*np.min(tau)) where tau is the intergrated autocorrelation
        time.
    :param verbose: (*bool, optional*)
        Output all the diagnostics? Defaults to False.

    :returns iburn: (*int*)
        burn-in index estimate.  If estBurnin == False, returns 0.
    :returns ithin: (*int*)
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