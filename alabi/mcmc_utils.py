from .mcmc import *

def runMCMC(self, samplerKwargs=None, mcmcKwargs=None, runName="apRun",
                cache=True, estBurnin=True, thinChains=True, verbose=False,
                args=None, **kwargs):
        """
        Given forward model input-output pairs, theta and y, and a trained GP,
        run an MCMC using the GP to evaluate the logprobability instead of the
        true, computationally-expensive forward model.

        Parameters
        ----------
        samplerKwargs : dict, optional
            Parameters for emcee.EnsembleSampler object
            If None, defaults to the following:
                nwalkers : int, optional
                    Number of emcee walkers.  Defaults to 10 * dim
        mcmcKwargs : dict, optional
            Parameters for emcee.EnsembleSampler.sample/.run_mcmc methods. If
            None, defaults to the following required parameters:
                iterations : int, optional
                    Number of MCMC steps.  Defaults to 10,000
                initial_state : array/emcee.State, optional
                    Initial guess for MCMC walkers.  Defaults to None and
                    creates guess from priors.
        runName : str, optional
            Filename prefix for all cached files, e.g. for hdf5 file where mcmc
            chains are saved.  Defaults to runNameii.h5. where ii is the
            current iteration number.
        cache : bool, optional
            Whether or not to cache MCMC chains, forward model input-output
            pairs, and GP kernel parameters.  Defaults to True since they're
            expensive to evaluate. In practice, users should cache forward model
            inputs, outputs, ancillary parameters, etc in each likelihood
            function evaluation, but saving theta and y here doesn't hurt.
            Saves the forward model, results to runNameAPFModelCache.npz,
            the chains as runNameii.h5 for each, iteration ii, and the GP
            parameters in runNameAPGP.npz in the current working directory, etc.
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
        args : iterable, optional
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict, optional
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.

        Returns
        -------
        sampler : emcee.EnsembleSampler
            emcee sampler object
        iburn : int
            burn-in index estimate.  If estBurnin == False, returns 0.
        ithin : int
            thin cadence estimate.  If thinChains == False, returns 1.
        """

        # Initialize, validate emcee.EnsembleSampler and run_mcmc parameters
        samplerKwargs, mcmcKwargs = mcmcUtils.validateMCMCKwargs(self,
                                                                 samplerKwargs,
                                                                 mcmcKwargs,
                                                                 verbose)

        # Create backend to save chains?
        if cache:
            bname = str(runName) + ".h5"
            self.backends.append(bname)
            backend = emcee.backends.HDFBackend(bname)
            backend.reset(samplerKwargs["nwalkers"], samplerKwargs["ndim"])
        # Only keep last sampler object in memory
        else:
            backend = None

        # Create sampler using GP lnlike function as forward model surrogate
        self.sampler = emcee.EnsembleSampler(**samplerKwargs,
                                             backend=backend,
                                             args=args,
                                             kwargs=kwargs,
                                             blobs_dtype=[("lnprior", float)])

        # Run MCMC!
        for _ in self.sampler.sample(**mcmcKwargs):
            pass
        if verbose:
            print("mcmc finished")

        # If estimating burn in or thin scale, compute integrated
        # autocorrelation length of the chains
        iburn, ithin = mcmcUtils.estimateBurnin(self.sampler,
                                                estBurnin=estBurnin,
                                                thinChains=thinChains,
                                                verbose=verbose)

        return self.sampler, iburn, ithin