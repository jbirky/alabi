# -*- coding: utf-8 -*-
"""
:py:mod:`mcmcUtils.py` - Markov Chain Monte Carlo Utility Functions
-------------------------------------------------------------------

MCMC utility functions for validating emcee MCMC runs within approxposterior.
"""

# Tell module what it's allowed to import
__all__ = ["validateMCMCKwargs", "batchMeansMCSE", "estimateBurnin"]

import numpy as np


def validateMCMCKwargs(ap, samplerKwargs, mcmcKwargs, verbose=False):
    """
    Validates emcee.EnsembleSampler parameters/kwargs.

    Parameters
    ----------
    ap : approxposterior.ApproxPosterior
            Initialized ApproxPosterior object
    samplerKwargs : dict
        dictionary containing parameters intended for emcee.EnsembleSampler
        object
    mcmcKwargs : dict
        dictionary containing parameters intended for
        emcee.EnsembleSampler.run_mcmc/.sample object
    verbose : bool, optional
        verboisty level. Defaults to False (no output)

    Returns
    -------
    samplerKwargs : dict
        Sanitized dictionary containing parameters intended for
        emcee.EnsembleSampler object
    mcmcKwargs : dict
        Sanitized dictionary containing parameters intended for
        emcee.EnsembleSampler.run_mcmc/.sample object
    """

    # First validate kwargs for emcee.EnsembleSampler object
    if samplerKwargs is None:
        samplerKwargs = dict()

        samplerKwargs["ndim"] = ap.ndim
        samplerKwargs["nwalkers"] = 20 * samplerKwargs["dim"]
        samplerKwargs["log_prob_fn"] = ap._gpll
    else:
        # If user set ndim, ignore it and align it with theta's dimensionality
        samplerKwargs.pop("ndim", None)
        samplerKwargs["ndim"] = ap.ndim

        # Initialize other parameters if they're not provided
        try:
            nwalkers = samplerKwargs["nwalkers"]
        except KeyError:
            print("WARNING: samplerKwargs provided but nwalkers not in samplerKwargs")
            print("Defaulting to nwalkers = 20 per dimension.")
            samplerKwargs["nwalkers"] = 20 * samplerKwargs["ndim"]

        if "backend" in samplerKwargs.keys():
            print("WARNING: backend in samplerKwargs. approxposterior creates its own!")
            print("with filename = apRun.h5. Disregarding user-supplied backend.")

        # Handle case when user supplies own loglikelihood function
        if "log_prob_fn" in samplerKwargs.keys():
            # Remove any other log_prob_fn
            samplerKwargs.pop("log_prob_fn", None)

        # Prevent users from providing their own backend
        samplerKwargs.pop("backend", None)

        # Properly initialize log_prob_fn to be GP loglikelihood estimate
        samplerKwargs["log_prob_fn"] = ap._gpll

    # Validate mcmcKwargs dict used in sampling posterior distribution
    # e.g. emcee.EnsembleSampler.run_mcmc method
    if mcmcKwargs is None:
        mcmcKwargs = dict()
        mcmcKwargs["iterations"] = 10000
        mcmcKwargs["initial_state"] = ap.priorSample(samplerKwargs["nwalkers"])
    else:
        try:
            nsteps = mcmcKwargs["iterations"]
        except KeyError:
            mcmcKwargs["iterations"] = 10000
            if verbose:
                print("WARNING: mcmcKwargs provided, but iterations not in mcmcKwargs.")
                print("Defaulting to iterations = 10000.")
        try:
            p0 = mcmcKwargs["initial_state"]
        except KeyError:
            mcmcKwargs["initial_state"] = ap.priorSample(samplerKwargs["nwalkers"])
            if verbose:
                print("WARNING: mcmcKwargs provided, but initial_state not in mcmcKwargs.")
                print("Defaulting to nwalkers samples from priorSample.")

    return samplerKwargs, mcmcKwargs
# end function


def batchMeansMCSE(samples, bins=None, fn=None):
    """
    Estimate the Monte Carlo Standard Error of MCMC samples using the
    non-overlapping batch means methods. See Flegal, Haran, & Jones (2008) for
    more info: https://arxiv.org/pdf/math/0703746.pdf

    Parameters
    ----------
    samples : array
        nsamples x ndim array of MCMC samples
    bins : int, optional
        Number of bins. Defaults to int(sqrt(len(samples)))
    fn : function, optional
        Function used to compute posterior summary statistic on each chunk.
        Defaults to None to compute the simple expected value, aka the mean.

    Returns
    -------
    RMSE : float/array
        RMSE for each dimension of the chain
    """

    # Initialize function if None provided
    if fn is None:
        fn = lambda x : x

    # Initialize number of chunks
    if bins is None:
        bins = max(int(np.sqrt(len(samples))),2)

    # num MUST be an interger
    assert isinstance(bins, int), "num must be an interger"

    # Figure out dimensionality
    samples = np.asarray(samples)
    ndim = samples.ndim

    # Compute b, init holder
    b = int(len(samples) / bins)
    if ndim > 1:
        ndim = samples.shape[-1]
        y = np.zeros((bins, ndim))
    else:
        y = np.zeros(bins)

    # Estimate summary statistic using entire chain
    mu = np.mean(fn(samples), axis=0)

    # Compute summary statistic for each bin in the chain
    for ii in range(bins):
        # Select current chunk, compute!
        lower = ii * b
        upper = (ii + 1) * b
        y[ii] = np.sum(fn(samples[lower:upper]), axis=0) / b

    # Estimate batch means (MCSE) as empirical standard deviation, return MCSE
    mcse = b / (bins - 1) * np.sum((y - mu)**2, axis=0)
    return np.sqrt(mcse / len(samples))
# end function


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
        print("thin estimate: %d" % ithin)

    return iburn, ithin
# end function
