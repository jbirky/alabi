# -*- coding: utf-8 -*-
"""
:py:mod:`core.py` - alabi
-------------------------------------

"""

__all__ = ["SurrogateModel"]

import utility as ut
import visualization as vis
import gp_utils
import mcmc_utils 
import cache_utils

import numpy as np
from scipy.optimize import minimize
from functools import partial
import george
from george import kernels
import multiprocessing as mp
import time
import emcee
import os
import warnings
import tqdm
import pickle


class SurrogateModel(object):

    def __init__(self, fn=None, bounds=None, labels=None, 
                 cache=True, savedir="results/", model_name="surrogate_model",
                 verbose=True, ncore=mp.cpu_count()):

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
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # Name of model cache
        self.model_name = model_name

        # Print progress statements
        self.verbose = verbose

        # Number of cores alabi is allowed to use
        if ncore > mp.cpu_count():
            self.ncore = mp.cpu_count()
        elif ncore <= 0:
            self.ncore = 1
        else:
            self.ncore = ncore

        # Names of input parameters, used for plots
        if labels is None:
            self.labels = [r"$\theta_%s$"%(i) for i in range(self.ndim)]
        else:
            self.labels = labels

        # false if emcee and dynesty have not been run for this object
        self.emcee_run = False
        self.dynesty_run = False

    
    def save(self):

        file = os.path.join(self.savedir, self.model_name)

        # pickle surrogate model object
        print(f"Caching model to {file}...")
        with open(file+".pkl", "wb") as f:        
            pickle.dump(self, f)

        cache_utils.write_report_gp(self, file)

        if self.emcee_run == True:
            cache_utils.write_report_emcee(self, file)

        if self.dynesty_run == True:
            cache_utils.write_report_dynesty(self, file)


    def init_train(self, nsample=None, sampler='sobol'):

        if nsample is None:
            nsample = 50 * self.ndim

        self.theta0 = ut.prior_sampler(nsample=nsample, bounds=self.bounds, sampler=sampler)
        self.y0 = ut.eval_fn(self.fn, self.theta0, ncore=self.ncore)

        self.theta = self.theta0
        self.y = self.y0

        # record number of training samples
        self.ninit_train = len(self.theta0)
        self.ntrain = self.ninit_train
        self.nactive = 0

        if self.cache:
            np.savez(f"{self.savedir}/initial_training_sample.npz", theta=self.theta, y=self.y)


    def init_test(self, nsample=None, sampler='sobol'):

        if nsample is None:
            nsample = 50 * self.ndim

        self.theta_test = ut.prior_sampler(nsample=nsample, bounds=self.bounds, sampler=sampler)
        self.y_test = ut.eval_fn(self.fn, self.theta_test, ncore=self.ncore)

        # record number of test samples
        self.ntest = len(self.theta_test)

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
            kernel = "ExpSquaredKernel"

        # Stationary kernels
        if kernel == "ExpSquaredKernel":
            # Guess initial metric, or scale length of the covariances (must be > 0)
            initial_lscale = np.fabs(np.random.randn(self.ndim))
            self.kernel = kernels.ExpSquaredKernel(metric=initial_lscale, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with squared exponential kernel.")
        elif kernel == "RationalQuadraticKernel":
            initial_lscale = np.fabs(np.random.randn(self.ndim))
            self.kernel = kernels.RationalQuadraticKernel(log_alpha=1, metric=initial_lscale, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with rational quadratic kernel.")
        elif kernel == "Matern32Kernel":
            initial_lscale = np.fabs(np.random.randn(self.ndim))
            self.kernel = kernels.Matern32Kernel(metric=initial_lscale, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with squared matern-3/2 kernel.")
        elif kernel == "Matern52Kernel":
            initial_lscale = np.fabs(np.random.randn(self.ndim))
            self.kernel = kernels.Matern52Kernel(metric=initial_lscale, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with squared matern-5/2 kernel.")

        # Non-stationary kernels
        elif kernel == "LinearKernel":
            self.kernel = kernels.LinearKernel(log_gamma2=1, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with linear kernel.")
        elif kernel == "ExpSine2Kernel":
            self.kernel = kernels.ExpSine2Kernel(gamma=1, log_period=1, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with exponential sin^2 kernel.")
        elif kernel == "CosineKernel":
            self.kernel = kernels.CosineKernel(log_period=1, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with cosine kernel.")
        elif kernel == "DotProductKernel":
            self.kernel = kernels.DotProductKernel(ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with dot product kernel.")
        elif kernel == "LocalGaussianKernel":
            center = np.median(self.bounds, axis=1)
            self.kernel = kernels.LocalGaussianKernel(location=center, log_width=1, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with local gaussian kernel.")
        elif kernel == "ConstantKernel":
            self.kernel = kernels.ConstantKernel(log_constant=1, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with constant kernel.")
        elif kernel == "PolynomialKernel":
            self.kernel = kernels.PolynomialKernel(log_sigma2=1, order=1, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with polynomial kernel.")

        # Custom george kernel
        else:
            try:
                self.kernel = kernel
                test_gp = george.GP(kernel)
                self.kernel_name = "Custom Kernel"
                print(f"Loaded custom kernel with parameters: {test_gp.get_parameter_names()}")
            except:
                print(f"kernel {kernel} is not valid.\n")
                print("Enter either one of the following options, or a george kernel object.")
                print(george.kernels.__all__)

        # assign GP hyperparameter prior
        self.gp_hyper_prior = gp_hyper_prior
        
        warnings.simplefilter("ignore")
        self.gp = gp_utils.fit_gp(self.theta, self.y, self.kernel, 
                                  fit_amp=fit_amp, fit_mean=fit_mean,
                                  white_noise=white_noise)

        t0 = time.time()
        self.gp = gp_utils.optimize_gp(self.gp, self.theta, self.y,
                                       gp_hyper_prior=self.gp_hyper_prior)
        tf = time.time()

        if self.verbose:
            print(f"optimized hyperparameters: ({np.round(tf - t0, 1)}s)")
            print(self.gp.get_parameter_vector())
            print('')


    def evaluate(self, theta):

        theta = np.asarray(theta).reshape(1,-1)

        return self.gp.predict(self.y, theta, return_cov=False)[0]


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

        # GP hyperparameter optimization frequency
        self.gp_opt_freq = gp_opt_freq

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
                                     "training_error" : [],
                                     "test_error" : [], 
                                     "gp_kl_divergence" : [],
                                     "gp_train_time" : [],
                                     "obj_fn_opt_time" : []}
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
                opt_obj_t0 = time.time()
                theta_new, y_new = self.find_next_point()
                opt_obj_tf = time.time()

                # add theta and y to training sample
                theta_prop = np.append(self.theta, [theta_new], axis=0)
                y_prop = np.append(self.y, y_new)

                try:
                    fit_gp_t0 = time.time()
                    # Fit GP. Make sure to feed in previous iteration hyperparameters!
                    gp = gp_utils.fit_gp(theta_prop, y_prop, self.kernel,
                                         hyperparameters=self.gp.get_parameter_vector())
                    fit_gp_tf = time.time()

                    # Optimize GP?
                    if ii % self.gp_opt_freq == 0:
                        t0 = time.time()
                        gp = gp_utils.optimize_gp(gp, theta_prop, y_prop,
                                                  gp_hyper_prior=self.gp_hyper_prior)
                        tf = time.time()
                        if self.verbose:
                            print(f"optimized hyperparameters: ({np.round(tf - t0, 1)}s)")
                            print(self.gp.get_parameter_vector())
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
            self.training_results["training_error"].append(training_error)
            self.training_results["test_error"].append(test_error)
            self.training_results["gp_kl_divergence"].append(gp_kl_divergence)
            self.training_results["gp_train_time"].append(fit_gp_tf - fit_gp_t0)
            self.training_results["obj_fn_opt_time"].append(opt_obj_tf - opt_obj_t0)

        # record total number of training samples
        self.ntrain = len(self.theta)
        # number of active training samples
        self.nactive = self.ntrain - self.ninit_train

        if self.cache:
            self.save()


    def train_gp_default(self):

        sm.init_train()
        sm.init_test()
        sm.init_gp()
        sm.active_train()


    def lnprob(self, theta):

        if not hasattr(self, 'gp'):
            raise NameError("GP has not been trained")

        if not hasattr(self, 'lnprior'):
            raise NameError("lnprior has not been specified")

        theta = np.asarray(theta).reshape(1,-1)

        return self.evaluate(theta) + self.lnprior(theta)


    def run_emcee(self, lnprior=None, nwalkers=None, nsteps=int(5e4),
                  opt_init=True, multi_proc=True):
        """
        Outputs:

        print   autocorrelation time
                acceptance fraction
                summary stats

        plot    walkers
                corner

        cache   pickle object with mcmc results
        """

        import emcee

        if lnprior is None:
            print(f"No lnprior specified. Defaulting to uniform prior with bounds {self.bounds}")
            self.lnprior = partial(ut.lnprior_uniform, bounds=self.bounds)
        else:
            self.lnprior = lnprior

        # number of walkers, and number of steps per walker
        if nwalkers is None:
            self.nwalkers = 10 * self.ndim
        else:
            self.nwalkers = nwalkers
        self.nsteps = nsteps

        if self.verbose:
            print(f"Running emcee with {nwalkers} walkers for {nsteps} steps...")

        # Optimize walker initialization?
        if opt_init == True:
            # start walkers near the estimated maximum
            p0 = find_map(lnprior=self.lnprior)
        else:
            # start walkers at random points in the prior space
            p0 = ut.prior_sampler(nsample=nwalkers, bounds=self.bounds)

        # set up multiprocessing pool
        if multi_proc == True:
            pool = mp.Pool(self.ncore)
        else:
            pool = None

        # set up hdf5 backend
        # if self.cache:
        #     backend = emcee.backends.HDFBackend(f"{self.savedir}/emcee_samples_raw.h5")
        #     backend.reset(nwalkers, self.ndim)
        # else:
        #     backend = None

        # Run the sampler!
        self.sampler = emcee.EnsembleSampler(self.nwalkers, 
                                             self.ndim, 
                                             self.lnprob, 
                                             pool=pool)

        self.sampler.run_mcmc(p0, self.nsteps, progress=True)

        # burn, thin, and flatten samples
        self.iburn, self.ithin = mcmc_utils.estimateBurnin(self.sampler, verbose=self.verbose)
        self.emcee_samples = self.sampler.get_chain(discard=self.iburn, flat=True, thin=self.ithin) 

        if self.verbose:
            # get acceptance fraction and autocorrelation time
            acc_frac = np.mean(self.sampler.acceptance_fraction)
            autcorr_time = np.mean(self.sampler.get_autocorr_time())

            print(f"Total samples: {self.emcee_samples.shape[0]}")
            print("Mean acceptance fraction: {0:.3f}".format(acc_frac))
            print("Mean autocorrelation time: {0:.3f} steps".format(autcorr_time))

        # record that emcee has been run
        self.emcee_run = True

        if self.cache:
            self.save()

            np.savez(f"{self.savedir}/emcee_samples_final.npz", samples=self.emcee_samples)

    
    def run_dynesty(self, ptform=None, multi_proc=True, sampler_kwargs={}):

        import dynesty
        from dynesty import DynamicNestedSampler
        from dynesty import utils as dyfunc

        # set up multiprocessing pool
        if multi_proc == True:
            pool = mp.Pool(self.ncore)
        else:
            pool = None

        # set up prior transform
        if ptform is None:
            self.ptform = partial(ut.prior_transform_uniform, bounds=self.bounds)
        else:
            self.ptform = ptform

        # initialize our nested sampler
        self.dsampler = DynamicNestedSampler(self.evaluate, 
                                             self.ptform, 
                                             self.ndim,
                                            #  pool=pool,
                                             **sampler_kwargs)

        self.dsampler.run_nested()
        self.res = self.dsampler.results

        samples = self.res.samples  # samples
        weights = np.exp(self.res.logwt - self.res.logz[-1])

        # Resample weighted samples.
        self.dynesty_samples = dyfunc.resample_equal(samples, weights)

        # record that dynesty has been run
        self.dynesty_run = True

        if self.cache:
            self.save()

            np.savez(f"{self.savedir}/dynesty_samples_final.npz", samples=self.dynesty_samples)


    def find_map(self, theta0=None, lnprior=None, method="nelder-mead", nRestarts=15, options=None):

        raise NotImplementedError("Not implemented.")


    def plot(self, plots=None, save=True):

        # ================================
        # GP training plots
        # ================================

        if "gp_all" in plots:
            gp_plots = ["gp_error", "gp_hyperparam", "gp_timing", "gp_train_corner"]
            for pl in gp_plots:
                plots.append(pl)

        # Test error vs iteration
        if "gp_error" in plots:
            if hasattr(self, "training_results"):
                print("Plotting gp error...")
                vis.plot_error_vs_iteration(self, log=True, title=f"{self.kernel_name} surrogate")
                vis.plot_error_vs_iteration(self, log=False, title=f"{self.kernel_name} surrogate")
            else:
                raise NameError("Must run active_train before plotting gp_error.")

        # GP hyperparameters vs iteration
        if "gp_hyperparam" in plots:
            if hasattr(self, "training_results"):
                print("Plotting gp hyperparameters...")
                vis.plot_hyperparam_vs_iteration(self, title=f"{self.kernel_name} surrogate")
            else:
                raise NameError("Must run active_train before plotting gp_hyperparam.")

        # GP training time vs iteration
        if "gp_timing" in plots:
            if hasattr(self, "training_results"):
                print("Plotting gp timing...")
                vis.plot_train_time_vs_iteration(self, title=f"{self.kernel_name} surrogate")
            else:
                raise NameError("Must run active_train before plotting gp_timing.")

        # N-D scatterplots and histograms colored by function value
        if "gp_train_corner" in plots:  
            if hasattr(self, "theta") and hasattr(self, "y"):
                print("Plotting training sample corner plot...")
                vis.plot_corner_scatter(self)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        # GP training time vs iteration
        if "gp_fit_2D" in plots:
            if hasattr(self, "theta") and hasattr(self, "y"):
                print("Plotting gp fit 2D...")
                if self.ndim == 2:
                    vis.plot_gp_fit_2D(self, ngrid=60, title=f"{self.kernel_name} surrogate")
                else:
                    print("theta must be 2D to use gp_fit_2D!")
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_fit_2D.")

        # ================================
        # emcee plots
        # ================================

        if "emcee_all" in plots:
            emcee_plots = ["emcee_corner", "emcee_walkers"]
            for pl in emcee_plots:
                plots.append(pl)

        # emcee posterior samples
        if "emcee_corner" in plots:  
            if hasattr(self, "emcee_samples"):
                print("Plotting emcee posterior...")
                vis.plot_corner(self, self.emcee_samples, sampler="emcee_")
            else:
                raise NameError("Must run run_emcee before plotting emcee_corner.")

        # emcee walkers
        if "emcee_walkers" in plots:  
            if hasattr(self, "emcee_samples"):
                print("Plotting emcee walkers...")
                vis.plot_emcee_walkers(self)
            else:
                raise NameError("Must run run_emcee before plotting emcee_walkers.")

        # ================================
        # dynesty plots
        # ================================

        if "dynesty_all" in plots:
            dynesty_plots = ["dynesty_corner", "dynesty_corner_kde", 
                             "dynesty_traceplot", "dynesty_runplot"]
            for pl in dynesty_plots:
                plots.append(pl)

        # dynesty posterior samples
        if "dynesty_corner" in plots:  
            if hasattr(self, "res"):
                print("Plotting dynesty posterior...")
                vis.plot_corner(self, self.dynesty_samples, sampler="dynesty_")
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_corner.")

        if "dynesty_corner_kde" in plots:  
            if hasattr(self, "dynesty_samples"):
                print("Plotting dynesty posterior kde...")
                vis.plot_corner_kde(self)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_corner.")

        if "dynesty_traceplot" in plots:
            if hasattr(self, "res"):
                print("Plotting dynesty traceplot...")
                vis.plot_dynesty_traceplot(self)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_traceplot.")

        if "dynesty_runplot" in plots:
            if hasattr(self, "res"):
                print("Plotting dynesty runplot...")
                vis.plot_dynesty_runplot(self)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_runplot.")

        # ================================
        # MCMC comparison plots
        # ================================

        if "mcmc_comparison" in plots:
            if hasattr(self, "emcee_samples") and hasattr(self, "res"):
                print("Plotting emcee vs dynesty posterior comparison...")
                vis.plot_mcmc_comparison(self)
            else:
                raise NameError("Must run run_emcee and run_dynesty before plotting emcee_comparison.")
                    