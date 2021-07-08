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
import pickle


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

    
    def save(self, fname="surrogate_model.pkl"):
        
        pkl_file = os.path.join(self.savedir, fname)
        print(f"Caching model to {pkl_file}...")
        with open(pkl_file, 'wb') as f:        
            pickle.dump(self, f)


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

        
    def init_gp(self, kernel=None, fit_amp=True, fit_mean=True, white_noise=-12, 
                gp_hyper_prior=None, overwrite=False):
        
        if hasattr(self, 'gp') and (overwrite == False):
            raise AssertionError("GP kernel already assigned. Use overwrite=True to re-assign the kernel.")
        
        # Initialize GP kernel
        if kernel is None:
            print("No kernel specified. Defaulting to squared exponential kernel.")
            kernel = 'sqexp'

        if kernel == 'sqexp':
            # Guess initial metric, or scale length of the covariances (must be > 0)
            initial_lscale = np.fabs(np.random.randn(self.ndim))
            self.kernel = george.kernels.ExpSquaredKernel(metric=initial_lscale, ndim=self.ndim)
            print("Initialized GP with squared exponential kernel.")
        elif kernel == 'matern32':
            initial_lscale = np.fabs(np.random.randn(self.ndim))
            self.kernel = george.kernels.Matern32Kernel(metric=initial_lscale, ndim=self.ndim)
            print("Initialized GP with squared matern-3/2 kernel.")
        elif kernel == 'matern52':
            initial_lscale = np.fabs(np.random.randn(self.ndim))
            self.kernel = george.kernels.Matern52Kernel(metric=initial_lscale, ndim=self.ndim)
            print("Initialized GP with squared matern-5/2 kernel.")
        else:
            # custom george kernel
            self.kernel = kernel

        # assign GP hyperparameter prior
        self.gp_hyper_prior = gp_hyper_prior
        
        warnings.simplefilter("ignore")
        self.gp = gp_utils.fit_gp(self.theta, self.y, self.kernel, 
                                  fit_amp=fit_amp, fit_mean=fit_mean,
                                  white_noise=white_noise)
        self.gp = gp_utils.optimize_gp(self.gp, self.theta, self.y,
                                       gp_hyper_prior=self.gp_hyper_prior)

        if self.verbose:
            print("optimized hyperparameters:", self.gp.get_parameter_vector())
            print('')


    def evaluate(self, theta, **kwargs):

        return self.gp.predict(self.y, theta, **kwargs)


    def find_next_point(self, theta0=None, nopt=5, **kwargs):
        
        # Find new theta that produces a valid loglikelihood
        thetaT, uT = ut.minimize_objective(self.utility, self.y, self.gp,
                                            bounds=self.bounds,
                                            nopt=nopt,
                                            args=(self.y, self.gp))

        # evaluate function at the optimized theta
        yT = self.fn(thetaT)

        return thetaT, yT


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

            # AGP on even, BAPE on odd
            if self.algorithm == "alternate":
                if ii % 2 == 0:
                    self.utility = ut.agp_utility
                else:
                    self.utility = ut.bape_utility

            while True:
                # Find next training point!
                theta_new, y_new = self.find_next_point()

                # add theta and y to training sample
                theta_prop = np.append(self.theta, [theta_new], axis=0)
                y_prop = np.append(self.y, y_new)

                try:
                    t0 = time.time()
                    # Fit GP. Make sure to feed in previous iteration hyperparameters!
                    gp = gp_utils.fit_gp(theta_prop, y_prop, self.kernel,
                                         hyperparameters=self.gp.get_parameter_vector())
                    tf = time.time()

                    # Optimize GP?
                    if ii % gp_opt_freq == 0:
                        gp = gp_utils.optimize_gp(gp, theta_prop, y_prop,
                                                  gp_hyper_prior=self.gp_hyper_prior,
                                                  p0=self.gp.get_parameter_vector())
                        if self.verbose:
                            print("optimized hyperparameters:", self.gp.get_parameter_vector())
                    break

                except:
                    if self.verbose:
                        msg = "Warning: GP fit failed. Likely covariance matrix was not positive definite. "
                        msg += "Attempting another training sample... "
                        msg += "If this issue persists, try expanding the hyperparameter prior"
                        print(msg)

            # If proposed (theta, y) did not cause fitting issues, save to surrogate model obj
            self.theta = theta_prop
            self.y = y_prop
            self.gp = gp

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
            self.training_results["gp_train_time"].append(tf - t0)
            self.training_results["training_error"].append(training_error)
            self.training_results["test_error"].append(test_error)
            self.training_results["gp_kl_divergence"].append(gp_kl_divergence)

        if self.cache:
            self.save()


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

        # Test error vs iteration
        if 'gp_error' in plots:
            if hasattr(self, 'training_results'):
                print("Plotting gp error...")

                vis.plot_error_vs_iteration(self.training_results, self.savedir, log=True)
                vis.plot_error_vs_iteration(self.training_results, self.savedir, log=False)
            else:
                raise NameError("Must run active_train before plotting gp_error.")

        # GP hyperparameters vs iteration
        if 'gp_hyperparam' in plots:
            if hasattr(self, 'training_results'):
                print("Plotting gp hyperparameters...")

                hp_names = self.gp.get_parameter_names()
                hp_values = np.array(self.training_results["gp_hyperparameters"])

                vis.plot_hyperparam_vs_iteration(self.training_results, hp_names, 
                                                 hp_values, self.savedir)
            else:
                raise NameError("Must run active_train before plotting gp_hyperparam.")

        # GP training time vs iteration
        if 'gp_train_time' in plots:
            if hasattr(self, 'training_results'):
                print("Plotting gp train time...")

                vis.plot_train_time_vs_iteration(self.training_results, self.savedir)
            else:
                raise NameError("Must run active_train before plotting gp_hyperparam.")

        # N-D scatterplots and histograms colored by function value
        if 'training_corner' in plots:  
            if hasattr(self, 'theta') and hasattr(self, 'y'):
                plot_corner_scatter(self.theta, self.y, self.labels, self.savedir)
            else:
                raise NameError("Must run init_train and/or active_train before plotting training_corner.")
                    