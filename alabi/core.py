# -*- coding: utf-8 -*-
"""
:py:mod:`core.py` - alabi
-------------------------------------

"""

__all__ = ["SurrogateModel"]

import utility as ut
import gp_utils
import visualization as vis

import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import math
import emcee
import corner
import george
import os
import warnings
import tqdm


class SurrogateModel(object):

    def __init__(self, fn=None, bounds=None, labels=None, 
                 cache=True, savedir='results/', verbose=True):

        # Check all required inputs are specified
        if fn is None:
            raise ValueError("Must supply fn to train GP surrogate model.")
        if bounds is None:
            raise ValueError("Must supply prior bounds.")

        # Set function for training the GP, and initial training samples
        # For bayesian inference problem this would be your log likelihood function
        self.fn = fn
        
        self.bounds = bounds

        # Determine dimensionality 
        self.ndim = len(self.bounds)

        # Cache surrogate model as pickle
        self.cache = cache 

        # Directory to save results and plots; defaults to local dir
        self.savedir = savedir
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        # Print progress statements
        self.verbose = verbose

        # Names of input parameters, used for plots
        if labels is None:
            self.labels = [r"$\theta_%s$"%(i) for i in range(self.ndim)]
        else:
            self.labels = labels


    def init_train(self, nsample=None, sampler='uniform', ncore=mp.cpu_count()):

        if nsample is None:
            nsample = 10 * self.ndim

        self.theta0 = ut.prior_sampler(nsample=nsample, bounds=self.bounds, sampler=sampler)
        self.y0 = ut.eval_fn(self.fn, self.theta0, ncore=ncore)

        self.theta = self.theta0
        self.y = self.y0

        if self.cache:
            np.savez(f"{self.savedir}/initial_training_sample.npz", theta=self.theta, y=self.y)


    def init_test(self, nsample=None, sampler='uniform', ncore=mp.cpu_count()):

        if nsample is None:
            nsample = 10 * self.ndim

        self.theta_test = ut.prior_sampler(nsample=nsample, bounds=self.bounds, sampler=sampler)
        self.y_test = ut.eval_fn(self.fn, self.theta_test, ncore=ncore)

        if self.cache:
            np.savez(f"{self.savedir}/initial_test_sample.npz", theta=self.theta_test, y=self.y_test)


    def load_train(self, savedir=None, theta0=None, y0=None):

        if savedir is not None:
            sims = np.load(f"{savedir}/initial_training_sample.npz")
            theta0 = sims["theta"]
            y0 = sims["y"]
        else:
            try:
                sims = np.load(f"{self.savedir}/initial_training_sample.npz")
                theta0 = sims["theta"]
                y0 = sims["y"]
            except:
                pass
            
        if (theta0 is not None) and (y0 is not None):
            self.theta0 = theta0 
            self.y0 = y0

        self.theta = self.theta0
        self.y = self.y0


    def load_train(self, savedir=None, theta_test=None, y_test=None):

        self.theta = theta_test 
        self.y = y_test

        
    def init_gp(self, kernel=None, fit_amp=True, fit_mean=True, white_noise=-15, overwrite=False):
        
        if hasattr(self, 'gp') and (overwrite == False):
            raise AssertionError("GP kernel already assigned. Use overwrite=True to re-assign the kernel.")
        
        # Initialize GP kernel
        if kernel is None:
            print("No kernel specified. Defaulting to squared exponential kernel.")

            # Guess initial metric, or scale length of the covariances (must be > 0)
            initial_lscale = np.fabs(np.random.randn(self.ndim))
            self.kernel = george.kernels.ExpSquaredKernel(metric=initial_lscale, ndim=self.ndim)
            
            if fit_amp == True:
                initial_amplitude = 1
                self.kernel *= initial_amplitude

        else:
            self.kernel = kernel
        
        self.gp = gp_utils.fit_gp(self.theta, self.y, self.kernel, 
                                  fit_amp=fit_amp, fit_mean=fit_mean,
                                  white_noise=white_noise)
        self.gp = gp_utils.optimize_gp(self.gp, self.theta, self.y)

        if self.verbose:
            print("\nInitialized GP.")
            print("optimized hyperparameters:", self.gp.get_parameter_vector())
            print('')


    def evaluate(self, theta, **kwargs):

        return gp.predict(self.y, theta, **kwargs)


    def find_next_point(self, theta0=None, nopt=5, **kwargs):
        
        # Find new theta that produces a valid loglikelihood
        thetaT, uT = ut.minimize_objective(self.utility, self.y, self.gp,
                                            bounds=self.bounds,
                                            nopt=nopt,
                                            args=(self.y, self.gp))

        # evaluate function at the optimized theta
        yT = self.fn(thetaT)

        # add theta and y to training sample
        self.theta = np.append(self.theta, [thetaT], axis=0)
        self.y = np.append(self.y, yT)


    def active_train(self, niter=100, algorithm="bape", gp_opt_freq=10): 

        # Set algorithm
        self.algorithm = str(algorithm).lower()

        # Assign utility function
        if self.algorithm == "bape":
            self.utility = ut.bape_utility
        elif self.algorithm == "agp":
            self.utility = ut.agp_utility
        elif self.algorithm == "alternate":
            # If alternate, AGP on even, BAPE on odd
            self.utility = ut.agp_utility
        elif self.algorithm == "jones":
            self.utility = ut.jones_utility
        else:
            errMsg = "Unknown algorithm. Valid options: bape, agp, jones, or alternate."
            raise ValueError(errMsg)

        if hasattr(self, 'training_results') == False:
            self.training_results = {"iteration" : [], 
                                     "gp_hyperparameters" : [], 
                                     "gp_train_time" : [], 
                                     "training_error" : [],
                                     "test_error" : [], 
                                     "gp_kl_divergence" : []}
            first_iter = 0
        else:
            first_iter = self.training_results["iteration"][-1]

        if self.verbose:
            print(f"Running {niter} active learning iterations using {self.algorithm}...")

        for ii in tqdm.tqdm(range(niter)):

            t0 = time.time()

            # AGP on even, BAPE on odd
            if self.algorithm == "alternate":
                if ii % 2 == 0:
                    self.utility = ut.agp_utility
                else:
                    self.utility = ut.bape_utility

            # Repeat active learning step until gp fit works w/o linear algebra errors
            gp_fit = True
            while gp_fit == True:
                try:
                    # Find m new (theta, y) pairs by maximizing utility function, one at a time
                    self.find_next_point()
                    print(theta.shape)

                    # Fit GP. Make sure to feed in previous iteration hyperparameters!
                    gp = gp_utils.fit_gp(self.theta, self.y, self.kernel,
                                         hyperparameters=self.gp.get_parameter_vector())
                    gp_fit = False 
                    if self.verbose:
                        print("Warning: GP fit failed. Likely non positive-definite GP covariance.")
                except:
                    gp_fit = True
            self.gp = gp

            # Optimize GP?
            if ii % gp_opt_freq == 0:
                self.gp = gp_utils.optimize_gp(self.gp, self.theta, self.y)
                if self.verbose:
                    print("optimized hyperparameters:", self.gp.get_parameter_vector())

            # evaluate gp training error
            ypred = self.gp.predict(self.y, self.theta, return_cov=False, return_var=False)
            training_error = np.mean((self.y - ypred)**2)

            # evaluate gp test error
            if hasattr(self, 'theta_test') and hasattr(self, 'y_test'):
                ytest = self.gp.predict(self.y, self.theta_test, return_cov=False, return_var=False)
                test_error = np.mean((self.y_test - ytest)**2)
            else:
                test_error = np.nan

            # evaluate convergence criteria
            gp_kl_divergence = np.nan

            # save results to a dictionary
            self.training_results["iteration"].append(ii + first_iter)
            self.training_results["gp_hyperparameters"].append(self.gp.get_parameter_vector())
            self.training_results["gp_train_time"].append(time.time() - t0)
            self.training_results["training_error"].append(training_error)
            self.training_results["test_error"].append(test_error)
            self.training_results["gp_kl_divergence"].append(gp_kl_divergence)


    def run_mcmc(self, sampler="emcee", lnprior=None, **kwargs):


        raise NotImplementedError("Not implemented.")

        import emcee

        if lnprior is None:
            def lnprior(theta):
                lnprior_uniform(theta, self.bounds)
        self.lnprior = lnprior

        def lnprob(theta):
            return self.lnprior(theta) + self.evaluate(theta)
        self.lnprob = lnprob

        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob)
        sampler.run_mcmc(p0, 10000)


    def find_map(self, theta0=None, lnprior=None, method="nelder-mead", nRestarts=15, options=None):

        raise NotImplementedError("Not implemented.")


    def plot(self, plots=None, save=True):

        if 'gp_error' in plots:
            if hasattr(self, 'training_results'):
                print("Plotting gp error...")

                # Test error vs iteration
                vis.plot_error_vs_iteration(self.training_results, self.savedir, log=True)
                vis.plot_error_vs_iteration(self.training_results, self.savedir, log=False)
            else:
                raise NameError("Must run active_train before plotting gp_error.")

        if 'gp_hyperparam' in plots:
            if hasattr(self, 'training_results'):
                print("Plotting gp hyperparameters...")

                # GP hyperparameters vs iteration
                hp_names = self.gp.get_parameter_names()
                hp_values = np.array(self.training_results["gp_hyperparameters"])

                vis.plot_hyperparam_vs_iteration(self.training_results, hp_names, 
                                                 hp_values, self.savedir)
            else:
                raise NameError("Must run active_train before plotting gp_hyperparam.")

        if 'training_corner' in plots:  
            if hasattr(self, 'theta') and hasattr(self, 'y'):
                plot_corner_scatter(self.theta, self.y, self.labels, self.savedir)
            else:
                raise NameError("Must run init_train and/or active_train before plotting training_corner.")
                    