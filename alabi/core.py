"""
:py:mod:`core.py` 
-------------------------------------
"""

__all__ = ["SurrogateModel"]

from alabi import utility as ut
from alabi import visualization as vis
from alabi import mcmc_utils 
from alabi import cache_utils
from alabi import metrics
from alabi import gp_utils

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
from functools import partial
import tqdm
import pickle
import time
import os
import multiprocessing as mp


class SurrogateModel(object):

    """
    Class used to estimate approximate Bayesian posterior distributions or perform Bayesian 
    optimization using a Gaussian process surrogate model

    :param fn: (*function, required*)
        Python function which takes input array ``theta`` and returns output array ``y=fn(theta)``. 
        For bayesian inference problems ``fn`` would be your log-likelihood function.

    :param bounds: (*array, required*)
        Prior bounds. List of min and max values for each dimension of ``theta``.
        Example: ``bounds = [(0,1), (2,3), ...]``

    :param labels: (*array, optional*)
    :param cache: (*bool, optional*)
    :param savedir: (*str, optional*)
    :param savedir: (*str, optional*)
    :param model_name: (*str, optional*)
    :param verbose: (*bool, optional*)
    :param ncore: (*int, optional*)
    :param scale: (*str, optional*)
            Method for scaling training sample ``y``. 
            Options: 
                - None -  no scaling
                - 'standard' - normalize by standard deviation
                - 'log'  - log scale 
                - 'nlog' - log negative scale (for probability distributions logP -> log(-logP)
    """

    def __init__(self, 
                 fn=None, 
                 bounds=None, 
                 labels=None, 
                 prior_sampler=None,
                 cache=True, 
                 summary=True,
                 savedir="results/", 
                 model_name="surrogate_model",
                 verbose=True, 
                 ncore=mp.cpu_count(), 
                 scale=None):

        # Check all required inputs are specified
        if fn is None:
            raise ValueError("Must supply fn to train GP surrogate model.")
        if bounds is None:
            raise ValueError("Must supply prior bounds.")
        
        # Print progress statements
        self.verbose = verbose
        
        # scale theta
        self.bounds = bounds
        # self.tscaler = MinMaxScaler()
        # self.tscaler.fit(jnp.array(self.bounds).T)

        # Set function for training the GP, and initial training samples
        # For bayesian inference problem this would be your log likelihood function
        # self.fn = fn
        self.scale = scale
        if self.verbose == True:
            if self.scale is None:
                print("No scaling applied to training sample.")
            else:
                print(f"Scaling function output by {scale}")
        self.fn = partial(self.scaled_function, fn=fn)

        if prior_sampler is None:
            self.prior_sampler = partial(ut.prior_sampler, bounds=self.bounds, sampler='sobol')
        else:
            self.prior_sampler = prior_sampler

        # Determine dimensionality 
        self.ndim = len(self.bounds)

        # Cache surrogate model as pickle
        self.cache = cache 

        # Save summary of model to text file
        self.summary = summary

        # Directory to save results and plots; defaults to local dir
        self.savedir = savedir
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # Name of model cache
        self.model_name = model_name

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
        self.numpyro_run = False

    
    def scaled_function(self, theta, fn):

        if self.scale is None:
            return fn(theta)
        elif self.scale.lower() == "log":
            return np.log10(fn(theta))
        elif self.scale.lower() == "nlog":
            return np.log10(-fn(theta))
        else:
            return fn(theta)

    
    def save(self):
        """
        Pickle ``SurrogateModel`` object and write summary to a text file
        """

        file = os.path.join(self.savedir, self.model_name)

        # pickle surrogate model object
        print(f"Caching model to {file}...")
        with open(file+".pkl", "wb") as f:        
            pickle.dump(self, f)

        if self.summary == True:
            if hasattr(self, "gp"):
                cache_utils.write_report_gp(self, file)

            if self.emcee_run == True:
                cache_utils.write_report_emcee(self, file)

            if self.dynesty_run == True:
                cache_utils.write_report_dynesty(self, file)


    def init_train(self, nsample=None, sampler="sobol"):
        """
        :param nsample: (*int, optional*) 
            Number of samples. Defaults to ``nsample = 50 * self.ndim``

        :param sampler: (*str, optional*) 
            Sampling method. Defaults to ``'sobol'``. 
            See ``utility.prior_sampler`` for more details.
        """

        if nsample is None:
            nsample = 50 * self.ndim

        self.theta0 = self.prior_sampler(nsample=nsample, sampler=sampler)
        self.y0 = ut.eval_fn(self.fn, self.theta0, ncore=self.ncore)

        self.theta = self.theta0
        self.y = self.y0

        if self.cache:
            np.savez(f"{self.savedir}/initial_training_sample.npz", theta=self.theta, y=self.y)


    def init_test(self, nsample=None):
        """
        :param nsample: (*int, optional*) 
            Number of samples. Defaults to ``nsample = 50 * self.ndim``

        :param sampler: (*str, optional*) 
            Sampling method. Defaults to ``'sobol'``. 
            See ``utility.prior_sampler`` for more details.
        """

        if nsample is None:
            nsample = 50 * self.ndim

        self.theta_test = self.prior_sampler(nsample=nsample)
        self.y_test = ut.eval_fn(self.fn, self.theta_test, ncore=self.ncore)

        if self.cache:
            np.savez(f"{self.savedir}/initial_test_sample.npz", theta=self.theta_test, y=self.y_test)


    def load_train(self, cache_file):

        print(f"Loading training sample from {cache_file}...")
        sims = np.load(cache_file)
        self.theta0 = sims["theta"]
        self.y0 = sims["y"]
        self.theta = self.theta0
        self.y = self.y0

        if self.ndim != self.theta.shape[1]:
            raise ValueError(f"Dimension of bounds (n={self.ndim}) does not \
                              match dimension of training theta (n={self.theta.shape[1]})")


    def load_test(self, cache_file):

        print(f"Loading test sample from {cache_file}...")
        sims = np.load(cache_file)
        self.theta_test = sims["theta"]
        self.y_test = sims["y"]

        if self.ndim != self.theta_test.shape[1]:
            raise ValueError(f"Dimension of bounds (n={self.ndim}) does not \
                              match dimension of test theta (n={self.theta_test.shape[1]})")


    def init_samples(self, 
                     train_file=None, 
                     test_file=None, 
                     reload=False,
                     ntrain=None, 
                     ntest=None, 
                     sampler="sobol"):
        """
        Draw set of initial training samples and test samples. 
        To load cached samples from a numpy zip file from a previous run, 
        you can specify the file using ``train_file`` or ``test_file``. 
        Otherwise to run new samples, you can specify the number of 
        training and test samples using ``ntrain`` and ``ntest``.

        :param train_file: (*str, optional*) 
            Path to cached training samples. E.g. ``train_file='results/initial_training_sample.npz'``

        :param test_file: (*str, optional*)
            Path to cached training samples. E.g. ``test_file='results/initial_test_sample.npz'``

        :param reload: (*bool, optional*)
            Attempt to load cached samples from default cache files? 
            Will not be used if user specifies ``train_file`` or ``test_file``. Defaults to False.

        :param ntrain: (*int, optional*)
            Number of training samples to compute.

        :param ntest: (*int, optional*)
            Number of test samples to compute.

        :param sampler: (*function, optional*)
            Prior function for drawing training samples. Defaults to uniform using self.bounds
        """

        if train_file is not None:
            self.load_train(f"{self.savedir}/{train_file}")  
        else:
            if reload == True:
                try:
                    cache_file = f"{self.savedir}/initial_training_sample.npz"
                    self.load_train(cache_file)
                except:
                    print(f"Unable to reload {cache_file}. Computing new samples...")
                    self.init_train(nsample=ntrain, sampler=sampler)
            else:
                self.init_train(nsample=ntrain, sampler=sampler)

        if test_file is not None:
            self.load_test(f"{self.savedir}/{test_file}")
        else:
            if reload == True:
                try:
                    cache_file = f"{self.savedir}/initial_test_sample.npz"
                    self.load_test(cache_file)
                except:
                    print(f"Unable to reload {cache_file}. Computing new samples...")
                    self.init_test(nsample=ntest)
            else:
                self.init_test(nsample=ntest)

        # convert to jnp arrays
        self.theta = jnp.array(self.theta)
        self.y = jnp.array(self.y)

        # record number of training samples
        self.ninit_train = len(self.theta0)
        self.ntrain = self.ninit_train
        self.nactive = 0

        # record number of test samples
        self.ntest = len(self.theta_test)
    

    def fit_gp(self, hyper_init=None, 
               theta=None, 
               y=None,
               opt_method="newton-cg"):

        if theta is None:
            theta = self.theta
        if y is None:
            y = self.y
        if hyper_init is None:
            hyper_init = self.hparam

        @jax.jit
        def gp_nll(hpar):
            return -gp_utils.build_gp(hpar, theta).log_probability(y)

        t0 = time.time()

        # Optimize GP hyperparameters
        solver = jaxopt.ScipyMinimize(fun=gp_nll, method=opt_method)
        self.hparam = solver.run(hyper_init).params
        self.gp = gp_utils.build_gp(self.hparam, theta).condition(y, theta).gp

        if self.verbose == True:
            print("Initial NLL:", gp_nll(hyper_init), "| Final NLL:", gp_nll(self.hparam))
        
        timing = time.time() - t0

        return self.gp, timing

        
    def init_gp(self, 
                fit_amp=True, 
                fit_mean=True, 
                fit_diag=True, 
                diag=-12.,
                hyper_init=None):
        
        self.kernel_name = "ExponentialSquared"
        
        if hyper_init == None:

            hyper_init = {"log_scale": np.zeros(self.ndim),
                          "log_amp": None,
                          "mean": None}

            if fit_mean == True:
                hyper_init["mean"] = np.mean(self.y)
            if fit_amp == True:
                hyper_init["log_amp"] = np.float64(0.0)
            # if fit_diag == True:
            #     hyper_init["log_diag"] = np.log(diag)

        self.fit_gp(hyper_init=hyper_init)


    def evaluate(self, theta_xs):
        """
        Evaluate predictive mean of the GP at point ``theta_xs``

        :param theta_xs: (*array, required*) Point to evaluate GP mean at.

        :returns ypred: (*float*) GP mean evaluated at ``theta_xs``.
        """

        theta_xs = np.asarray(theta_xs).reshape(1,-1)

        # apply the GP
        ypred = self.gp.predict(self.y, theta_xs, return_cov=False)[0]

        if self.scale == "log":
            ypred = 10**ypred 
        elif self.scale == "nlog":
            ypred = -10**ypred 

        return ypred


    def neg_evaluate(self, theta_xs):

        return -1 * self.evaluate(theta_xs)
    

    def active_learning_objective(self, xs):

        # bound_prior = ut.lnprior_uniform(xs, self.bounds)

        mu, var = self.gp.predict(self.y, xs.reshape(1,-1), return_var=True)

        if self.algorithm == "bape":
            util = -((2.0 * mu + var) + jnp.log(jnp.exp(var) - 1))[0]
        elif self.algorithm == "agp":
            util = -(mu + 0.5*jnp.log(2.0*jnp.pi*jnp.e*var))[0]
        else:
            raise KeyError(f"algorithm='{self.algorithm}' is not a valid option.")
            
        return util 


    def find_next_point(self):
        """
        Find next set of ``(theta, y)`` training points by maximizing the
        active learning utility function.

        :param nopt: (*int, optional*) 
            Number of times to restart the objective function optimization. 
            Defaults to 1. Increase to avoid converging to local maxima.
        """

        opt_timing_0 = time.time()

        theta_init = self.prior_sampler(nsample=1, sampler="uniform")

        solver = jaxopt.ScipyBoundedMinimize(fun=self.active_learning_objective)
        thetaN = solver.run(theta_init, bounds=jnp.array(self.bounds).T).params.flatten()

        opt_timing = time.time() - opt_timing_0

        # evaluate function at the optimized theta
        yN = self.fn(thetaN)

        return thetaN, yN, opt_timing


    def active_train(self, 
                     niter=100, 
                     algorithm="bape", 
                     gp_save_freq=10, 
                     plots=None,
                     obj_opt_method="nelder-mead",
                     max_integration_pts=int(10**5)): 
        """
        :param niter: (*int, optional*)

        :param algorithm: (*str, optional*) 

        :param gp_save_freq: (*int, optional*)

        :param save_progress: (*bool, optional*)
        """

        # Set algorithm
        self.algorithm = str(algorithm).lower()

        # GP hyperparameter optimization frequency
        self.gp_save_freq = gp_save_freq

        # Objective function optimization method
        self.obj_opt_method = obj_opt_method

        if hasattr(self, 'training_results') == False:
            self.training_results = {"iteration" : [], 
                                     "gp_hyperparameters" : [],  
                                     "training_error" : [],
                                     "test_error" : [], 
                                     "gp_kl_divergence" : [],
                                     "gp_js_divergence" : [],
                                     "gp_train_time" : [],
                                     "obj_fn_opt_time" : []}
            first_iter = 0
        else:
            first_iter = self.training_results["iteration"][-1]

        if self.verbose:
            print(f"Running {niter} active learning iterations using {self.algorithm}...")

        # start timing active learning 
        train_t0 = time.time()

        # save previous iteration gp
        self.prev_gp = self.gp

        # array of integration points for computing kl divergence
        dx = ut.prior_sampler(bounds=self.bounds, 
                              nsample=min(int(10**self.ndim), max_integration_pts), 
                              sampler="sobol")

        for ii in tqdm.tqdm(range(1, niter+1)):

            # Find next training point!
            thetaN, yN, opt_timing = self.find_next_point()

            # Reshape the array to shape (1, N)
            theta_reshaped = jnp.reshape(thetaN, (1, thetaN.shape[0]))

            # add theta and y to training sample
            self.theta = jnp.append(self.theta, theta_reshaped, axis=0)
            self.y = jnp.append(self.y, yN)

            if self.verbose == True:
                print("Total training samples:", len(self.theta))

            # fit GP to new training sample
            self.gp, fit_gp_timing = self.fit_gp()

            # record active learning train runtime
            self.train_runtime = time.time() - train_t0 

            # evaluate gp training error (scaled)
            ypred = self.gp.predict(self.y, self.theta, return_cov=False, return_var=False)
            training_error = np.mean((self.y - ypred)**2)

            # evaluate gp test error (scaled)
            if hasattr(self, 'theta_test') and hasattr(self, 'y_test'):
                if (len(self.theta_test) > 0):
                    ytest = self.gp.predict(self.y, self.theta_test, return_cov=False, return_var=False)
                    test_error = np.mean((self.y_test - ytest)**2)
                else:
                    test_error = np.nan
            else:
                test_error = np.nan

            # evaluate convergence criteria
            yint_prev = self.prev_gp.predict(self.y[:-1], dx, return_cov=False, return_var=False)
            yint = self.gp.predict(self.y, dx, return_cov=False, return_var=False)
            gp_kl_divergence = metrics.kl_divergence(yint_prev, yint)
            gp_js_divergence = metrics.js_divergence(yint_prev, yint)

            # save results to a dictionary
            self.training_results["iteration"].append(ii + first_iter)
            self.training_results["gp_hyperparameters"].append(self.hparam)
            self.training_results["training_error"].append(training_error)
            self.training_results["test_error"].append(test_error)
            self.training_results["gp_kl_divergence"].append(gp_kl_divergence)
            self.training_results["gp_js_divergence"].append(gp_js_divergence)
            self.training_results["gp_train_time"].append(fit_gp_timing)
            self.training_results["obj_fn_opt_time"].append(opt_timing)


            # record total number of training samples
            self.ntrain = len(self.theta)
            # number of active training samples
            self.nactive = self.ntrain - self.ninit_train

            # save gp for the next iteration
            self.prev_gp = self.gp

            # Optimize GP?
            if (ii + first_iter) % self.gp_save_freq == 0:
                
                if ii != 0:
                    if plots is not None:
                        self.plot(plots=plots)
                    else:
                        self.plot(plots=["gp_error", "gp_hyperparam"])
                        if self.ndim == 1:
                            self.plot(plots=["gp_fit_1d"])
                        elif self.ndim == 2:
                            self.plot(plots=["gp_fit_2d"])
                        else:
                            self.plot(plots=["gp_train_scatter"])

        if self.cache:
            self.save()


    def lnprob(self, theta):
        """
        Log probability function used for ``emcee``, which sums the prior with the surrogate model likelihood

        .. math::

            \\ln P(\\theta | x) \\propto \\ln P(x | \\theta) + \\ln P(\\theta)

        where \\ln P(x | \\theta) is the surrogate likelihood function and \\ln P(\\theta) is the prior function.

        :param theta: (*array, required*) 
            Array of model input parameters to evaluate model probability at.
        """

        if not hasattr(self, "lnprior"):
            raise NameError("lnprior has not been specified")

        if self.like_name.lower() != "true":
            theta = np.asarray(theta).reshape(1,-1)

        lnp = self.like_fn(theta) + self.lnprior(theta)

        return lnp


    def find_map(self, theta0=None, lnprior=None, method="nelder-mead", nRestarts=15, options=None):

        raise NotImplementedError("Not implemented.")


    def run_emcee(self, 
                  like_fn="surrogate", 
                  lnprior=None, 
                  nwalkers=None, 
                  nsteps=int(5e4), 
                  p0=None,
                  sampler_kwargs={}, 
                  run_kwargs={},
                  opt_init=False, 
                  multi_proc=True, 
                  lnprior_comment=None):
        """
        Use the ``emcee`` affine-invariant MCMC package to sample the trained GP surrogate model.
        https://github.com/dfm/emcee

        :param lnprior: (*function, optional*) 
            Log-prior function.
            Defaults to uniform prior using the function ``utility.lnprior_uniform`` 
            with bounds ``self.bounds``.

        :param nwalkers: (*int, optional*) 
            Number of MCMC walkers. Defaults to ``self.nwalkers = 10 * self.ndim``.

        :param nsteps: (*int, optional*) 
            Number of steps per walker. Defaults to ``nsteps=int(5e4)``.

        :param sampler_kwargs: (*dict, optional*) 

        :param run_kwargs: (*dict, optional*) 

        :param opt_init: (*bool, optional*) 

        :param multi_proc: (*bool, optional*) 

        :param lnprior_comment: (*str, optional*) 
        """
        import emcee

        # specify likelihood function (true function or surrogate model)
        if like_fn.lower() == "true":
            print("Initializing emcee with self.fn as likelihood.")
            self.like_fn = self.fn
        else:
            print("Initializing emcee with self.evaluate surrogate model as likelihood.")
            if not hasattr(self, "gp"):
                raise NameError("GP has not been trained")
            else:
                self.like_fn = self.evaluate
        self.like_name = like_fn

        if lnprior is None:
            print(f"No lnprior specified. Defaulting to uniform prior with bounds {self.bounds}")
            self.lnprior = partial(ut.lnprior_uniform, bounds=self.bounds)

            # Comment for output log file
            self.lnprior_comment =  f"Default uniform prior. \n" 
            self.lnprior_comment += f"Prior function: ut.lnprior_uniform\n"
            self.lnprior_comment += f"\twith bounds {self.bounds}"

        else:
            self.lnprior = lnprior

            # Comment for output log file
            if lnprior_comment is None:
                self.lnprior_comment = f"User defined prior."
                try:
                    self.lnprior_comment += f"Prior function: {self.lnprior.__name__}"
                except:
                    self.lnprior_comment += "Prior function: unrecorded"
            else:
                self.lnprior_comment = lnprior_comment

        # number of walkers, and number of steps per walker
        if nwalkers is None:
            self.nwalkers = 10 * self.ndim
        else:
            self.nwalkers = nwalkers
        self.nsteps = nsteps

        if self.verbose:
            print(f"Running emcee with {nwalkers} walkers for {nsteps} steps...")

        # Optimize walker initialization?
        if p0 is None:
            if opt_init == True:
                # start walkers near the estimated maximum
                p0 = self.find_map(lnprior=self.lnprior)
            else:
                # start walkers at random points in the prior space
                p0 = ut.prior_sampler(nsample=nwalkers, bounds=self.bounds)

        # set up multiprocessing pool
        if multi_proc == True:
            pool = mp.Pool(self.ncore)
        else:
            pool = None

        # Run the sampler!
        emcee_t0 = time.time()
        self.sampler = emcee.EnsembleSampler(self.nwalkers, 
                                             self.ndim, 
                                             self.lnprob, 
                                             pool=pool,
                                             **sampler_kwargs)

        self.sampler.run_mcmc(p0, self.nsteps, progress=True, **run_kwargs)

        # record emcee runtime
        self.emcee_runtime = time.time() - emcee_t0

        # burn, thin, and flatten samples
        self.iburn, self.ithin = mcmc_utils.estimate_burnin(self.sampler, verbose=self.verbose)
        self.emcee_samples_full = self.sampler.get_chain()
        self.emcee_samples = self.sampler.get_chain(discard=self.iburn, flat=True, thin=self.ithin) 

        # get acceptance fraction and autocorrelation time
        self.acc_frac = np.mean(self.sampler.acceptance_fraction)
        self.autcorr_time = np.mean(self.sampler.get_autocorr_time())
        if self.verbose:
            print(f"Total samples: {self.emcee_samples.shape[0]}")
            print("Mean acceptance fraction: {0:.3f}".format(self.acc_frac))
            print("Mean autocorrelation time: {0:.3f} steps".format(self.autcorr_time))

        # record that emcee has been run
        self.emcee_run = True

        # close pool
        if pool is not None:
            pool.close()

        if self.cache:
            self.save()

            np.savez(f"{self.savedir}/emcee_samples_final.npz", samples=self.emcee_samples)

    
    def run_dynesty(self, 
                    like_fn="surrogate", 
                    ptform=None, 
                    mode="dynamic", 
                    sampler_kwargs={}, 
                    run_kwargs={},
                    multi_proc=False, 
                    save_iter=None, 
                    ptform_comment=None):
        """
        Use the ``dynesty`` nested-sampling MCMC package to sample the trained GP surrogate model.
        https://github.com/joshspeagle/dynesty

        :param ptform: (*function, optional*) 
            Log-prior transform function.
            Defaults to uniform prior using the function ``utility.prior_transform_uniform`` 
            with bounds ``self.bounds``.

        :param sampler_kwargs: (*dict, optional*) 

        :param run_kwargs: (*dict, optional*) 

        :param multi_proc: (*bool, optional*) 

        :param ptform_comment: (*str, optional*) 
        """

        from dynesty import NestedSampler
        from dynesty import DynamicNestedSampler
        from dynesty import utils as dyfunc

        # set up multiprocessing pool
        if multi_proc == True:
            pool = mp.Pool(self.ncore)
            pool.size = self.ncore
        else:
            pool = None

        # specify likelihood function (true function or surrogate model)
        if like_fn.lower() == "true":
            print("Initializing dynesty with self.fn as likelihood.")
            self.like_fn = self.fn
        else:
            print("Initializing dynesty with self.evaluate surrogate model as likelihood.")
            if not hasattr(self, "gp"):
                raise NameError("GP has not been trained")
            else:
                self.like_fn = self.evaluate
        self.like_name = like_fn

        # set up prior transform
        if ptform is None:
            self.ptform = partial(ut.prior_transform_uniform, bounds=self.bounds)

            # Comment for output log file
            self.ptform_comment =  f"Default uniform prior transform. \n" 
            self.ptform_comment += f"Prior function: ut.prior_transform_uniform\n"
            self.ptform_comment += f"\twith bounds {self.bounds}"
        
        else:
            self.ptform = ptform

            # Comment for output log file
            if ptform_comment is None:
                self.ptform_comment = f"User defined prior transform."
                try:
                    self.ptform_comment += f"Prior function: {self.ptform.__name__}"
                except:
                    self.ptform_comment += "Prior function: unrecorded"
            else:
                self.ptform_comment = ptform_comment

        # start timing dynesty
        dynesty_t0 = time.time()

        # initialize our nested sampler
        if mode == 'dynamic':
            dsampler = DynamicNestedSampler(self.like_fn, 
                                            self.ptform, 
                                            self.ndim,
                                            pool=pool,
                                            **sampler_kwargs)
            print("Initialized dynesty DynamicNestedSampler.")
        elif mode == 'static':
            dsampler = NestedSampler(self.like_fn, 
                                     self.ptform, 
                                     self.ndim,
                                     pool=pool,
                                     **sampler_kwargs)
            print("Initialized dynesty NestedSampler.")
        else:
            raise ValueError(f"mode {mode} is not a valid option. Choose 'dynamic' or 'static'.")

        # Pickle sampler?
        if save_iter is not None:
            run_sampler = True
            last_iter = 0
            while run_sampler == True:
                dsampler.run_nested(maxiter=save_iter, **run_kwargs)
                self.res = dsampler.results

                file = os.path.join(self.savedir, "dynesty_sampler.pkl")

                # pickle dynesty sampler object
                print(f"Caching model to {file}...")
                with open(file, "wb") as f:        
                    pickle.dump(dsampler, f)

                # check if converged (i.e. hasn't run for more iterations)
                if dsampler.results.niter > last_iter:
                    last_iter = dsampler.results.niter
                    run_sampler = True
                else:
                    run_sampler = False
        else:
            dsampler.run_nested(**run_kwargs)

        # record dynesty runtime
        self.dynesty_runtime = time.time() - dynesty_t0

        self.res = dsampler.results
        samples = self.res.samples  # samples
        weights = np.exp(self.res.logwt - self.res.logz[-1])

        # Resample weighted samples.
        self.dynesty_samples = dyfunc.resample_equal(samples, weights)

        # record that dynesty has been run
        self.dynesty_run = True

        # close pool
        if pool is not None:
            pool.close()

        if self.cache:
            self.save()

            np.savez(f"{self.savedir}/dynesty_samples_final.npz", samples=self.dynesty_samples)


    def run_numpyro(self):

        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        raise NotImplementedError("Not implemented.")


    def plot(self, plots=None):
        """
        Plot diagnostics for training sample, GP fit, MCMC, etc.

        :param plots: 
            List of plots to generate. Will return an exception if ``SurrogateModel`` 
            has not run the function which creates the data used for the plot. Options:
                ``'gp_error'``
                ``'gp_hyperparam'``
                ``'gp_timing'``
                ``'gp_train_corner'``
                ``'gp_train_iteration'``
                ``'gp_fit_1d'``
                ``'gp_fit_2d'``
                ``'emcee_corner'``
                ``'emcee_walkers'``
                ``'dynesty_corner'``
                ``'dynesty_corner_kde'``
                ``'dynesty_traceplot'``
                ``'dynesty_runplot'``
                ``'mcmc_comparison'``
        :type plots: array, required
        """

        # ================================
        # GP training plots
        # ================================

        # make strings not case sensitive
        plots = [s.lower() for s in plots]

        if "gp_all" in plots:
            gp_plots = ["gp_error", "gp_hyperparam", "gp_timing", "gp_train_scatter"]
            if self.ndim == 2:
                gp_plots.append("gp_fit_2D")
            for pl in gp_plots:
                plots.append(pl)

        # Test error vs iteration
        if "gp_error" in plots:
            if hasattr(self, "training_results"):
                if self.verbose == True:
                    print("Plotting gp error...")
                vis.plot_error_vs_iteration(self, log=True, title=f"{self.kernel_name} surrogate")
                vis.plot_error_vs_iteration(self, log=False, title=f"{self.kernel_name} surrogate")
            else:
                raise NameError("Must run active_train before plotting gp_error.")

        # GP hyperparameters vs iteration
        if "gp_hyperparam" in plots:
            if hasattr(self, "training_results"):
                if self.verbose == True:
                    print("Plotting gp hyperparameters...")
                vis.plot_hyperparam_vs_iteration(self, title=f"{self.kernel_name} surrogate")
            else:
                raise NameError("Must run active_train before plotting gp_hyperparam.")

        # GP training time vs iteration
        if "gp_timing" in plots:
            if hasattr(self, "training_results"):
                if self.verbose == True:
                    print("Plotting gp timing...")
                vis.plot_train_time_vs_iteration(self, title=f"{self.kernel_name} surrogate")
            else:
                raise NameError("Must run active_train before plotting gp_timing.")

        # N-D scatterplots and histograms colored by function value
        if "gp_train_corner" in plots:  
            if hasattr(self, "theta") and hasattr(self, "y"):
                if self.verbose == True:
                    print("Plotting training sample corner plot...")
                vis.plot_corner_lnp(self)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        # N-D scatterplots and histograms
        if "gp_train_scatter" in plots:  
            if hasattr(self, "theta") and hasattr(self, "y"):
                if self.verbose == True:
                    print("Plotting training sample corner plot...")
                vis.plot_corner_scatter(self)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        if "gp_train_iteration" in plots:  
            if hasattr(self, "theta") and hasattr(self, "y"):
                if self.verbose == True:
                    print("Plotting training sample corner plot...")
                vis.plot_train_sample_vs_iteration(self)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_iteration.")

        if "gp_fit_1d" in plots:
            if hasattr(self, "theta") and hasattr(self, "y"):
                if self.verbose == True:
                    print("Plotting gp fit 1D...")
                if self.ndim == 1:
                    vis.plot_gp_fit_1D(self, ngrid=60, title=f"{self.kernel_name} surrogate")
                else:
                    print("theta must be 1D to use gp_fit_1d!")
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_fit_1d.")
            
        if "gp_fit_2d" in plots:
            if hasattr(self, "theta") and hasattr(self, "y"):
                if self.verbose == True:
                    print("Plotting gp fit 2D...")
                if self.ndim == 2:
                    vis.plot_gp_fit_2D(self, ngrid=60, title=f"{self.kernel_name} surrogate")
                else:
                    print("theta must be 2D to use gp_fit_2d!")
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_fit_2d.")

        # Objective function contour plot
        if "obj_fn_2d" in plots:
            if hasattr(self, "theta") and hasattr(self, "y") and hasattr(self, "gp"):
                if self.verbose == True:
                    print("Plotting objective function contours 2D...")
                vis.plot_utility_2D(self, ngrid=60)
            else:
                raise NameError("Must run init_train and init_gp before plotting obj_fn_2D.")

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
                if self.verbose == True:
                    print("Plotting emcee posterior...")
                vis.plot_corner(self, self.emcee_samples, sampler="emcee_")
            else:
                raise NameError("Must run run_emcee before plotting emcee_corner.")

        # emcee walkers
        if "emcee_walkers" in plots:  
            if hasattr(self, "emcee_samples"):
                if self.verbose == True:
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
                if self.verbose == True:
                    print("Plotting dynesty posterior...")
                vis.plot_corner(self, self.dynesty_samples, sampler="dynesty_")
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_corner.")

        if "dynesty_corner_kde" in plots:  
            if hasattr(self, "dynesty_samples"):
                if self.verbose == True:
                    print("Plotting dynesty posterior kde...")
                vis.plot_corner_kde(self)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_corner.")

        if "dynesty_traceplot" in plots:
            if hasattr(self, "res"):
                if self.verbose == True:
                    print("Plotting dynesty traceplot...")
                vis.plot_dynesty_traceplot(self)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_traceplot.")

        if "dynesty_runplot" in plots:
            if hasattr(self, "res"):
                if self.verbose == True:
                    print("Plotting dynesty runplot...")
                vis.plot_dynesty_runplot(self)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_runplot.")

        # ================================
        # MCMC comparison plots
        # ================================

        if "mcmc_comparison" in plots:
            if hasattr(self, "emcee_samples") and hasattr(self, "res"):
                if self.verbose == True:
                    print("Plotting emcee vs dynesty posterior comparison...")
                vis.plot_mcmc_comparison(self)
            else:
                raise NameError("Must run run_emcee and run_dynesty before plotting emcee_comparison.")
                    