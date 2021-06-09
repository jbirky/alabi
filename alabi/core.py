# -*- coding: utf-8 -*-
"""
:py:mod:`approx.py` - ApproxPosterior
-------------------------------------

Approximate Bayesian Posterior estimation and Bayesian optimzation. approxposterior
uses Dan Forman-Mackey's Gaussian Process implementation, george, and the
Metropolis-Hastings MCMC ensemble sampler, emcee, to infer the approximate
posterior distributions given the GP model.

"""

# Tell module what it's allowed to import
__all__ = ["SurrogateModel"]

from . import utility as ut
from . import gpUtils
from . import mcmcUtils
from . import visualization

import numpy as np
from scipy.optimize import minimize
import time
import emcee
import george
import os
import warnings
import tqdm


class SurrogateModel(object):
    """
    Class used to estimate approximate Bayesian posterior distributions or
    perform Bayesian optimization using a Gaussian process surrogate model

    Initial parameters:

    Parameters
    ----------
    fn : function
        Defines the log likelihood function.  In this function, it is assumed
        that the forward model is evaluated on the input theta and the output
        is used to evaluate the log likelihood.
    bounds : tuple/iterable
        Hard bounds for parameters
    gp : george.GP, optional
        Gaussian Process that learns the likelihood conditioned on forward
        model input-output pairs (theta, y). It's recommended that users
        specify their own kernel, GP using george. If None is provided, then
        approxposterior initialized a GP with a single ExpSquaredKernel as
        these work well in practice.
    theta : array-like, optional
        Input features (n_samples x n_features).  Defaults to None.
    y : array-like, optional
        Input result of forward model (n_samples,). Defaults to None.
    algorithm : str, optional
        Point selection algorithm that specifies which utility (also
        referred to as acquisition) function to use.  Defaults to bape.
        Options are bape (Bayesian Active Learning for Posterior Estimation,
        Kandasamy et al. (2015)), agp (Adapted Gaussian Process Approximation,
        Wang & Li (2017)), alternate (between AGP and BAPE), and jones
        (Jones et al. (1998) expected improvement).
        Case doesn't matter. If alternate, runs agp on even numbers and bape
        on odd.

        For approximate Bayesian posterior estimation, bape or alternate
        are typically the best optimizations. For Bayesian optimization,
        jones (expected improvement) usually performs best.
    cache : bool, optional
    savedir : str, optional

    Returns
    -------
    """

    def __init__(self, fn=None, bounds=None, 
                 gp=None, theta=[], y=[],  
                 algorithm="bape", cache=True, savedir='.'):
        """
        Initializer.
        """

        # Check all required inputs are specified
        if fn is None:
            raise ValueError("Must supply fn or lnlike to train GP on.")
        if bounds is None:
            raise ValueError("Must supply bounds for prior.")

        # Set function for training the GP, and initial training samples
        self.fn = fn

        # Make sure y, theta are valid 
        if (len(theta) != 0) and (len(y) != 0):
            if len(theta) != len(y):
                raise ValueError("theta and y must be the same length.")
            else:
                self.theta = np.array(theta)
                self.y = np.array(y)
        else:
            self.theta = np.array([])
            self.y = np.array([])

        if np.any(~np.isfinite(self.theta)) or np.any(~np.isfinite(self.y)):
            print("theta, y:", theta, y)
            raise ValueError("All theta and y values must be finite!")

        # Determine dimensionality 
        self.ndim = len(bounds)

        # Check the number of parameters in theta matches the number of bounds
        if (len(theta) > 0) and (theta.shape[-1] != ndim):
            err_msg = "ERROR: bounds provided but len(bounds) != ndim.\n"
            err_msg += "ndim = %d, len(bounds) = %d" % (ndim, len(bounds))
            raise ValueError(err_msg)
        else:
            self.bounds = bounds

        # Set algorithm
        self.algorithm = str(algorithm).lower()

        # Assign utility function
        if self.algorithm == "bape":
            self.utility = ut.BAPEUtility
        elif self.algorithm == "agp":
            self.utility = ut.AGPUtility
        elif self.algorithm == "alternate":
            # If alternate, AGP on even, BAPE on odd
            self.utility = ut.AGPUtility
        elif self.algorithm == "jones":
            self.utility = ut.JonesUtility
        else:
            errMsg = "Unknown algorithm. Valid options: bape, agp, naive, or alternate."
            raise ValueError(errMsg)

        # Initialize gaussian process if none provided
        if gp is None:
            print("INFO: No GP specified. Initializing GP using ExpSquaredKernel.")
            self.gp = gpUtils.defaultGP(self.theta, self.y)
        else:
            self.gp = gp

        # Cache surrogate model as pickle
        self.cache = cache 

        # Directory to save results and plots; defaults to local dir
        self.savedir = savedir


    def evaluate(self, theta, *args, **kwargs):
        """
        Compute the approximate posterior conditional distibution, the
        likelihood + prior learned by the GP, at a given point, theta.

        Parameters
        ----------
        theta : array-like
            Test point to evaluate GP posterior conditional distribution

        Returns
        -------
        mu : float
            Mean of predicted GP conditional posterior estimate at theta
        lnprior : float
            log prior evlatuated at theta
        """

        # Sometimes the input values can be crazy and the GP will blow up
        if not np.any(np.isfinite(theta)):
            return -np.inf, np.nan

        # Reject point if prior forbids it
        lnprior = self._lnprior(theta)
        if not np.isfinite(lnprior):
            return -np.inf, np.nan

        # Mean of predictive distribution conditioned on y (GP posterior estimate)
        # and make sure theta is the right shape for the GP
        try:
            mu = self.gp.predict(self.y, np.array(theta).reshape(1,-1),
                                 return_cov=False,
                                 return_var=False)
        except ValueError:
            return -np.inf, np.nan

        # Catch NaNs/Infs because they can (rarely) happen
        if not np.isfinite(mu):
            return -np.inf, np.nan
        else:
            return mu, lnprior


    def optGP(self, seed=None, method="powell", options=None, p0=None,
              nGPRestarts=1, gpHyperPrior=gpUtils.defaultHyperPrior):
        """
        Optimize hyperparameters of approx object's GP

        Parameters
        ----------
        seed : int, optional
            numpy RNG seed.  Defaults to None.
        nGPRestarts : int, optional
            Number of times to restart GP hyperparameter optimization.  Defaults
            to 1. Increase this number if the GP is not well-optimized.
        method : str, optional
            scipy.optimize.minimize method.  Defaults to powell.
        options : dict, optional
            kwargs for the scipy.optimize.minimize function.  Defaults to None.
        p0 : array, optional
            Initial guess for kernel hyperparameters.  If None, defaults to
            np.random.randn for each parameter
        gpHyperPrior : str/callable, optional
            Prior function for GP hyperparameters. Defaults to the defaultHyperPrior fn.
            This function asserts that the mean must be negative and that each log
            hyperparameter is within the range [-20,20].

        Returns
        -------
        optimizedGP : george.GP
        """

        # Optimize and reasign gp
        self.gp = gpUtils.optimizeGP(self.gp, self.theta, self.y, seed=seed,
                                     method=method, options=options,
                                     p0=p0, nGPRestarts=nGPRestarts,
                                     gpHyperPrior=gpHyperPrior)

    
    def find_next_point(self, theta0=None, computeLnLike=True, seed=None,
                      cache=True, gpOptions=None, gpP0=None, verbose=True,
                      nGPRestarts=1, nMinObjRestarts=5, gpMethod="powell",
                      minObjMethod="nelder-mead", minObjOptions=None,
                      runName="apRun", numNewPoints=1, optGPEveryN=1,
                      gpHyperPrior=gpUtils.defaultHyperPrior, args=None,
                      **kwargs):
                      
        return NotImplementedError("Not implemented.")


    def initial_train(self, ninit=None, sampler='uniform'):

        return NotImplementedError("Not implemented.")


    def active_train(self, niter=None):

        # implement default number of iterations
        if niter is None:
            niter = 100  

        # for nn in range(niter):
        #     if verbose:
        #         print("Iteration: %d" % nn)

        #     if timing:
        #         start = time.time()

        #     # 1) Find m new (theta, y) pairs by maximizing utility function,
        #     # one at a time. Note that computeLnLike = True means new points are
        #     # saved in self.theta, and self.y, expanding the training set
        #     # 2) In this function, GP hyperparameters are reoptimized after every
        #     # optGPEveryN new points
        #     _, _ = self.find_next_point(computeLnLike=True,
        #                               seed=seed,
        #                               cache=cache,
        #                               gpMethod=gpMethod,
        #                               gpOptions=gpOptions,
        #                               nGPRestarts=nGPRestarts,
        #                               nMinObjRestarts=nMinObjRestarts,
        #                               optGPEveryN=optGPEveryN,
        #                               numNewPoints=m,
        #                               gpHyperPrior=gpHyperPrior,
        #                               minObjMethod=minObjMethod,
        #                               minObjOptions=minObjOptions,
        #                               runName=runName,
        #                               theta0=None, # Sample from prior
        #                               args=args,
        #                               **kwargs)

        return NotImplementedError("Not implemented.")


    def run_mcmc(self, sampler="emcee", **kwargs):

        return NotImplementedError("Not implemented.")


    def find_map(self, theta0=None, method="nelder-mead", options=None,
                nRestarts=15):

        return MAP, -MAPVal


    def bayesOpt(self, nmax, theta0=None, tol=1.0e-3, kmax=3, seed=None,
                 verbose=True, runName="apRun", cache=True, gpMethod="powell",
                 gpOptions=None, gpP0=None, optGPEveryN=1, nGPRestarts=1,
                 nMinObjRestarts=5, initGPOpt=True, minObjMethod="nelder-mead",
                 gpHyperPrior=gpUtils.defaultHyperPrior, minObjOptions=None,
                 find_map=True, args=None, **kwargs):

        return soln