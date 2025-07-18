"""
:py:mod:`core.py` 
-------------------------------------
"""

__all__ = ["SurrogateModel"]

from alabi import utility as ut
from alabi import visualization as vis
from alabi import gp_utils
from alabi import mcmc_utils 
from alabi import cache_utils

import numpy as np
from scipy.optimize import minimize
from sklearn import preprocessing
from skopt.space import Real
from functools import partial
import george
from george import kernels
import multiprocessing as mp
import time
import os
import warnings
import tqdm
import pickle


# Define scaling functions 
def nlog(x): return np.log10(-x)
def nlog_inv(x): return -10**x
def log_scale(x): return np.log10(x)
def log_scale_inv(logx): return 10**logx
def no_scale(x): return x


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

    def __init__(self, fn=None, bounds=None, labels=None, 
                 cache=True, savedir="results/", model_name="surrogate_model",
                 verbose=True, ncore=mp.cpu_count(), scale=None, ignore_warnings=True):

        # Check all required inputs are specified
        if fn is None:
            raise ValueError("Must supply fn to train GP surrogate model.")
        if bounds is None:
            raise ValueError("Must supply prior bounds.")


        # Set function for training the GP, and initial training samples
        # For bayesian inference problem this would be your log likelihood function
        self.fn = fn
        self.scale = scale
        
        # unscaled bounds for theta 
        self.bounds = np.array(bounds)
        
        # Scale inputs between 0 and 1
        self.theta_scaler = preprocessing.MinMaxScaler()
        self.theta_scaler.fit(self.bounds.T)
        
        # Scale bounds to [0, 1] for training
        self._bounds = self.theta_scaler.transform(self.bounds)
        
        if scale == "nlog":
            # If scale is nlog, define the nlog scaler
            self.y_scaler = preprocessing.FunctionTransformer(func=nlog, inverse_func=nlog_inv)
        elif scale == "log":
            # If scale is log, define the log scaler
            self.y_scaler = preprocessing.FunctionTransformer(func=log_scale, inverse_func=log_scale_inv)
        elif scale == "minmax":
            self.y_scaler = preprocessing.MinMaxScaler()
        else:
            self.y_scaler = preprocessing.FunctionTransformer(func=no_scale, inverse_func=no_scale)

        # define prior sampler with scaled and unscaled bounds 
        self._prior_sampler = partial(ut.prior_sampler, bounds=self._bounds)
        self.prior_sampler = partial(ut.prior_sampler, bounds=self.bounds)

        # Determine dimensionality 
        self.ndim = len(self._bounds)

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
        
        # Ignore warnings
        if ignore_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

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
        """
        Pickle ``SurrogateModel`` object and write summary to a text file
        """

        file = os.path.join(self.savedir, self.model_name)

        # pickle surrogate model object
        print(f"Caching model to {file}...")
        with open(file+".pkl", "wb") as f:        
            pickle.dump(self, f)

        if hasattr(self, "gp"):
            cache_utils.write_report_gp(self, file)

        if self.emcee_run == True:
            cache_utils.write_report_emcee(self, file)

        if self.dynesty_run == True:
            cache_utils.write_report_dynesty(self, file)
            
    def _fn(self, _theta):
        """
        Internal function to evaluate the model function ``fn`` at scaled theta.
        This is used to avoid scaling the theta in the main function call.
        """
        # Unscale theta
        theta = self.theta_scaler.inverse_transform(_theta).flatten()
        
        # Evaluate function
        y = self.fn(theta)

        # Scale y
        _y = self.y_scaler.transform(y.reshape(-1, 1)).flatten()

        return _y
            
    def theta(self):
        
        return self.theta_scaler.inverse_transform(self._theta)
    
    def y(self):
        
        return self.y_scaler.inverse_transform(self._y.reshape(-1, 1)).flatten()

    def init_train(self, nsample=None, sampler="uniform", fname="initial_training_sample.npz"):
        """
        :param nsample: (*int, optional*) 
            Number of samples. Defaults to ``nsample = 50 * self.ndim``

        :param sampler: (*str, optional*) 
            Sampling method. Defaults to ``'sobol'``. 
            See ``utility.prior_sampler`` for more details.
        """

        if nsample is None:
            nsample = 50 * self.ndim

        _theta = self._prior_sampler(nsample=nsample, sampler=sampler)
        theta = self.theta_scaler.inverse_transform(_theta)
        
        y = ut.eval_fn(self.fn, theta, ncore=self.ncore).reshape(-1, 1)
        _y = self.y_scaler.fit_transform(y).flatten()

        if self.cache:
            np.savez(f"{self.savedir}/{fname}", theta=theta, y=y)

        return _theta, _y


    def load_train(self, cache_file):

        sims = np.load(cache_file)
        theta = sims["theta"]
        y = sims["y"]
        
        _theta = self.theta_scaler.transform(theta)
        _y = self.y_scaler.transform(y.reshape(-1, 1)).flatten()

        if self.ndim != theta.shape[1]:
            raise ValueError(f"Dimension of bounds (n={self.ndim}) does not \
                              match dimension of training theta (n={theta.shape[1]})")

        return _theta, _y


    def init_samples(self, train_file=None, test_file=None, reload=False,
                     ntrain=None, ntest=None, sampler="uniform"):
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

        # Load or create training sample
        if reload == True:
            try:
                cache_file = f"{self.savedir}/initial_training_sample.npz"
                print(f"Loading training sample from {cache_file}...")
                _theta, _y = self.load_train(cache_file)
                
            except:
                print(f"Unable to reload {cache_file}. Computing new samples...")
                _theta, _y = self.init_train(nsample=ntrain, sampler=sampler, fname="initial_training_sample.npz")
        else:
            _theta, _y = self.init_train(nsample=ntrain, sampler=sampler)

        # Load or create test sample
        if reload == True:
            try:
                cache_file = f"{self.savedir}/initial_test_sample.npz"
                print(f"Loading test sample from {cache_file}...")
                _theta_test, _y_test = self.load_test(cache_file)
            except:
                print(f"Unable to reload {cache_file}. Computing new samples...")
                _theta_test, _y_test = self.init_train(nsample=ntest, sampler=sampler, fname="initial_test_sample.npz")
        else:
            _theta_test, _y_test = self.init_train(nsample=ntest, sampler=sampler, fname="initial_test_sample.npz")

        # Training dataset scaled
        self._theta0 = _theta
        self._theta = _theta
        self._y = _y
        
        # Test dataset scaled
        self._theta_test = _theta_test
        self._y_test = _y_test

        # record number of training samples
        self.ninit_train = len(self._theta0)
        self.ntrain = self.ninit_train
        self.nactive = 0

        # record number of test samples
        self.ntest = len(self._theta_test)


    def set_hyperparam_prior_bounds(self):

        # Configure GP hyperparameter prior
        hp_bounds = self.gp.get_parameter_bounds()
        pnames = self.gp.get_parameter_names(include_frozen=False)
        
        if self.fit_mean:
            mean_bounds = [np.mean(self._y) - np.std(self._y),
                           np.mean(self._y) + np.std(self._y)]
            hp_bounds[pnames.index("mean:value")] = mean_bounds
            
        if self.fit_amp:
            amp_bounds = [0.1, 10]
            hp_bounds[pnames.index("kernel:k1:log_constant")] = amp_bounds

        if self.fit_white_noise:
            wn_bounds = [self.white_noise - 3, self.white_noise + 3]
            hp_bounds[pnames.index("white_noise:value")] = wn_bounds

        self.hp_bounds = np.array(hp_bounds)
        self.gp_hyper_prior = partial(ut.lnprior_uniform, bounds=self.hp_bounds)

    
    def fit_gp(self, theta=None, y=None):

        if theta is None:
            theta = self._theta

        if y is None:
            y = self._y

        t0 = time.time()

        self.set_hyperparam_prior_bounds()
        gp = gp_utils.configure_gp(theta, y, self.kernel, 
                                    fit_amp=self.fit_amp, 
                                    fit_mean=self.fit_mean,
                                    fit_white_noise=self.fit_white_noise,
                                    white_noise=self.white_noise)
        if gp is None:
            print(f"Warning: fit_gp failed with point {theta[-1]}. Reoptimizing hyperparameters...")
            gp, _ = self.opt_gp()
            
        gp.compute(theta)
        
        timing = time.time() - t0

        return gp, timing


    def opt_gp(self):

        t0 = time.time()

        self.set_hyperparam_prior_bounds()
        
        failed = True

        if failed == True:
            # create array of random initial hyperparameters:
            p0 = ut.prior_sampler(bounds=self.hp_bounds, nsample=self.gp_nopt, sampler='uniform')
            if hasattr(self, "gp"):
                # if gp exists, use current hyperparameters as a starting point 
                current_hp = self.gp.get_parameter_vector(include_frozen=False)
                if np.isfinite(self.gp_hyper_prior(current_hp)):
                    p0[0] = current_hp

            op_gp = gp_utils.optimize_gp(self.gp, self._theta, self._y,
                                        self.gp_hyper_prior, p0,
                                        method=self.gp_opt_method,
                                        bounds=self.hp_bounds)
            if op_gp is None:
                print("Warning: opt_gp hyperparameter optimization failed. Reoptimizing hyperparameters...")
                # op_gp = self.gp
                failed = True 
            else:
                failed = False

        tf = time.time()
        timing = tf - t0

        if self.verbose:
            print(f"Optimized hyperparameters: ({np.round(timing, 1)}s)")
            print(self.gp.get_parameter_names(include_frozen=False))
            print(self.gp.get_parameter_vector(include_frozen=False))
            print('')

        return op_gp, timing

        
    def init_gp(self, 
                kernel="ExpSquaredKernel",
                fit_amp=True, 
                fit_mean=True, 
                fit_white_noise=True, 
                white_noise=-12, 
                gp_scale_rng=1,
                overwrite=False,
                gp_opt_method="nelder-mead", 
                gp_nopt=3):
        """
        Initialize the Gaussian Process with specified kernel.

        :param kernel: (*str/george kernel obj, optional*) 
            ``george`` kernel object. Defaults to "ExpSquaredKernel". 
            See https://george.readthedocs.io/en/latest/user/kernels/ for more details.
            Options:
                ``'ExpSquaredKernel'``,
                ``'Matern32Kernel'``,
                ``'Matern52Kernel'``,
                ``'RationalQuadraticKernel'``
        """
        
        # optional hyperparameter choices
        self.fit_amp = fit_amp
        self.fit_mean = fit_mean
        self.fit_white_noise = fit_white_noise
        self.white_noise = white_noise

        # GP hyperparameter optimization method
        self.gp_opt_method = gp_opt_method

        # GP hyperparameter number of opt restarts
        self.gp_nopt = gp_nopt

        # assign GP hyperparameter prior
        self.gp_scale_rng = gp_scale_rng
        
        # set the bounds for scale length parameters
        metric_bounds = [(-self.gp_scale_rng, self.gp_scale_rng) for _ in range(self.ndim)]
        
        # theta ranges between 0 and 1, so choose an initial scale length between [0,1] 
        initial_lscale = np.random.uniform(0, 1, self.ndim)
        
        if hasattr(self, 'gp') and (overwrite == False):
            raise AssertionError("GP kernel already assigned. Use overwrite=True to re-assign the kernel.")
        
        # Initialize GP kernel
        # Stationary kernels
        if kernel == "ExpSquaredKernel":
            # Guess initial metric, or scale length of the covariances (must be > 0)
            self.kernel = kernels.ExpSquaredKernel(metric=initial_lscale, metric_bounds=metric_bounds, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with squared exponential kernel.")
        elif kernel == "RationalQuadraticKernel":
            self.kernel = kernels.RationalQuadraticKernel(log_alpha=1, metric=initial_lscale, metric_bounds=metric_bounds, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with rational quadratic kernel.")
        elif kernel == "Matern32Kernel":
            self.kernel = kernels.Matern32Kernel(metric=initial_lscale, metric_bounds=metric_bounds, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with squared matern-3/2 kernel.")
        elif kernel == "Matern52Kernel":
            self.kernel = kernels.Matern52Kernel(metric=initial_lscale, metric_bounds=metric_bounds, ndim=self.ndim)
            self.kernel_name = kernel
            print("Initialized GP with squared matern-5/2 kernel.")

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
                
        # set bounds for optional hyperparameters
        self.gp_nparam = self.ndim
        if self.fit_mean:
            self.gp_nparam += 1
        if self.fit_amp:
            self.gp_nparam += 1
        if self.fit_white_noise:
            self.gp_nparam += 1
            
        # create GP first time 
        self.gp = gp_utils.configure_gp(self._theta, self._y, self.kernel, 
                                        fit_amp=self.fit_amp, 
                                        fit_mean=self.fit_mean,
                                        fit_white_noise=self.fit_white_noise,
                                        white_noise=self.white_noise)
        
        # Fit GP with training sample and kernel
        self.gp, _ = self.fit_gp()

        # Optimize GP hyperparameters
        self.gp, _ = self.opt_gp()


    def evaluate(self, theta_xs):
        """
        Evaluate predictive mean of the GP at point ``theta_xs``

        :param theta_xs: (*array, required*) Point to evaluate GP mean at.

        :returns ypred: (*float*) GP mean evaluated at ``theta_xs``.
        """

        theta_xs = np.asarray(theta_xs).reshape(1,-1)
        _theta_xs = self.theta_scaler.transform(theta_xs)

        # apply the GP
        _ypred = self.gp.predict(self._y, _theta_xs, return_cov=False)[0]
        ypred = self.y_scaler.inverse_transform(_ypred.reshape(-1, 1)).flatten()

        return ypred


    def neg_evaluate(self, theta_xs):

        return -1 * self.evaluate(theta_xs)


    def find_next_point(self, nopt=3, opt_init=False):
        """
        Find next set of ``(theta, y)`` training points by maximizing the
        active learning utility function.

        :param nopt: (*int, optional*) 
            Number of times to restart the objective function optimization. 
            Defaults to 1. Increase to avoid converging to local maxima.
        """

        if opt_init:
            p0 = self._prior_sampler(nsample=1)
            t0 = minimize(self.neg_evaluate, p0, bounds=tuple(self._bounds), method="nelder-mead")["x"]
        else:
            t0 = None

        opt_timing_0 = time.time()
            
        _thetaN, _ = ut.minimize_objective(self.utility, self._y, self.gp,
                                          bounds=self._bounds,
                                          nopt=nopt,
                                          t0=t0,
                                          ps=self._prior_sampler,
                                          args=(self._y, self.gp, self._bounds),
                                          method=self.obj_opt_method)

        opt_timing = time.time() - opt_timing_0

        # evaluate function at the optimized theta
        _thetaN = _thetaN.reshape(1, -1)
        _yN = self._fn(_thetaN)

        return _thetaN, _yN, opt_timing


    def active_train(self, niter=100, algorithm="bape", gp_opt_freq=10, save_progress=False,
                     opt_init=False, obj_opt_method="nelder-mead"): 
        """
        :param niter: (*int, optional*)

        :param algorithm: (*str, optional*) 

        :param gp_opt_freq: (*int, optional*)

        :param save_progress: (*bool, optional*)
        """

        # Set algorithm
        self.algorithm = str(algorithm).lower()
        self.utility = ut.assign_utility(self.algorithm)

        # GP hyperparameter optimization frequency
        self.gp_opt_freq = gp_opt_freq

        # Objective function optimization method
        self.obj_opt_method = obj_opt_method

        if hasattr(self, 'training_results') == False:
            self.training_results = {"iteration" : [], 
                                     "gp_hyperparameters" : [],  
                                     "training_mse" : [],
                                     "test_mse" : [], 
                                     "training_scaled_mse" : [],
                                     "test_scaled_mse" : [],
                                     "gp_kl_divergence" : [],
                                     "gp_train_time" : [],
                                     "obj_fn_opt_time" : []}
            first_iter = 0
        else:
            first_iter = self.training_results["iteration"][-1]

        if self.verbose:
            print(f"Running {niter} active learning iterations using {self.algorithm}...")

        # start timing active learning 
        train_t0 = time.time()

        for ii in tqdm.tqdm(range(1, niter+1)):

            # AGP on even, BAPE on odd
            if self.algorithm == "alternate":
                if ii % 2 == 0:
                    self.utility = ut.agp_utility
                else:
                    self.utility = ut.bape_utility

            # Find next training point!
            thetaN, yN, opt_timing = self.find_next_point(opt_init=opt_init)
            
            # add theta and y to training samples
            theta_prop = np.append(self._theta, thetaN, axis=0)
            y_prop = np.append(self._y, yN)

            # Fit GP with new training point
            self.gp, fit_gp_timing = self.fit_gp(theta=theta_prop, y=y_prop)
            
            # If proposed (theta, y) did not cause fitting issues, save to surrogate model obj
            self._theta = theta_prop
            self._y = y_prop

            print("Total training samples:", len(self._theta))

            # record active learning train runtime
            self.train_runtime = time.time() - train_t0 

            # evaluate gp training error (scaled)
            _ypred = self.gp.predict(self._y, self._theta, return_cov=False, return_var=False)
            ypred = self.y_scaler.inverse_transform(_ypred.reshape(-1, 1)).flatten()
            training_mse = np.mean((self.y() - ypred)**2)
            training_scaled_mse = training_mse / np.var(self.y())

            # evaluate gp test error (scaled)
            if hasattr(self, '_theta_test') and hasattr(self, '_y_test'):
                if (len(self._theta_test) > 0):
                    _ytest = self.gp.predict(self._y, self._theta_test, return_cov=False, return_var=False)
                    ytest = self.y_scaler.inverse_transform(_ytest.reshape(-1, 1)).flatten()
                    ytest_true = self.y_scaler.inverse_transform(self._y_test.reshape(-1, 1)).flatten()
                    test_mse = np.mean((ytest_true - ytest)**2)
                    test_scaled_mse = test_mse / np.var(self.y())
                else:
                    test_mse = np.nan
            else:
                test_mse = np.nan

            # evaluate convergence criteria
            gp_kl_divergence = np.nan

            # save results to a dictionary
            self.training_results["iteration"].append(ii + first_iter)
            self.training_results["gp_hyperparameters"].append(self.gp.get_parameter_vector())
            self.training_results["training_mse"].append(training_mse)
            self.training_results["test_mse"].append(test_mse)
            self.training_results["training_scaled_mse"].append(training_scaled_mse)
            self.training_results["test_scaled_mse"].append(test_scaled_mse)
            self.training_results["gp_kl_divergence"].append(gp_kl_divergence)
            self.training_results["gp_train_time"].append(fit_gp_timing)
            self.training_results["obj_fn_opt_time"].append(opt_timing)

            # record total number of training samples
            self.ntrain = len(self._theta)
            # number of active training samples
            self.nactive = self.ntrain - self.ninit_train

            # Optimize GP?
            if (ii + first_iter) % self.gp_opt_freq == 0:

                # re-optimize hyperparamters
                self.gp, _ = self.opt_gp()
                
                if (save_progress == True) and (ii != 0):
                    self.save()
                    self.plot(plots=["gp_error", "gp_hyperparam"])
                    if self.ndim == 2:
                        self.plot(plots=["gp_fit_2D"])
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

        if not hasattr(self, 'gp'):
            raise NameError("GP has not been trained")

        if not hasattr(self, 'lnprior'):
            raise NameError("lnprior has not been specified")
        
        if not hasattr(self, 'like_fn'):
            self.like_fn = self.evaluate

        theta = np.asarray(theta).reshape(1,-1)

        lnp = self.like_fn(theta) + self.lnprior(theta)

        return lnp


    def find_map(self, theta0=None, lnprior=None, method="nelder-mead", nRestarts=15, options=None):

        raise NotImplementedError("Not implemented.")


    def run_emcee(self, like_fn="surrogate", lnprior=None, nwalkers=None, nsteps=int(5e4), sampler_kwargs={}, run_kwargs={},
                  opt_init=False, multi_proc=True, lnprior_comment=None):
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
            self.like_fn = self.evaluate

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
            self.nwalkers = int(10 * self.ndim)
        else:
            self.nwalkers = int(nwalkers)
        self.nsteps = int(nsteps)

        if self.verbose:
            print(f"Running emcee with {self.nwalkers} walkers for {self.nsteps} steps...")

        # Optimize walker initialization?
        if opt_init == True:
            # start walkers near the estimated maximum
            p0 = self.find_map(lnprior=self.lnprior)
        else:
            # start walkers at random points in the prior space
            p0 = ut.prior_sampler(nsample=self.nwalkers, bounds=self._bounds, sampler="uniform")

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

    
    def run_dynesty(self, like_fn="surrogate", ptform=None, mode="dynamic", sampler_kwargs={}, run_kwargs={},
                    multi_proc=False, save_iter=None, ptform_comment=None):
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

        import dynesty
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
            self.like_fn = self.evaluate

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


    def plot(self, plots=None, save=True, show=False):
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
                ``'gp_fit_2D'``
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

        if "gp_all" in plots:
            gp_plots = ["gp_error", "gp_hyperparam", "gp_timing", "gp_train_scatter"]
            if self.ndim == 2:
                gp_plots.append("gp_fit_2D")
            for pl in gp_plots:
                plots.append(pl)

        # Test error vs iteration
        if "gp_error" in plots:
            if hasattr(self, "training_results"):
                print("Plotting gp error...")
                
                iarray = np.array(self.training_results["iteration"])
                
                # MSE 
                vis.plot_error_vs_iteration(iarray,
                                            self.training_results["training_mse"],
                                            self.training_results["test_mse"],
                                            metric="Mean Squared Error",
                                            log=False,
                                            title=f"{self.kernel_name} surrogate",
                                            savedir=self.savedir,
                                            savename="gp_mse_vs_iteration.png",
                                            show=show)
                
                # Scaled MSE
                vis.plot_error_vs_iteration(iarray,
                                            self.training_results["training_scaled_mse"],
                                            self.training_results["test_scaled_mse"],
                                            metric="Mean Squared Error / Variance",
                                            log=False,
                                            title=f"{self.kernel_name} surrogate",
                                            savedir=self.savedir,
                                            savename="gp_scaled_mse_vs_iteration.png",
                                            show=show)
                
                # Log MSE
                vis.plot_error_vs_iteration(iarray,
                                            self.training_results["training_mse"],
                                            self.training_results["test_mse"],
                                            metric="Log(Mean Squared Error)",
                                            log=True,
                                            title=f"{self.kernel_name} surrogate",
                                            savedir=self.savedir,
                                            savename="gp_mse_vs_iteration_log.png",
                                            show=show)
            else:
                raise NameError("Must run active_train before plotting gp_error.")


        # GP hyperparameters vs iteration
        if "gp_hyperparam" in plots:
            if hasattr(self, "training_results"):
                print("Plotting gp hyperparameters...")
                vis.plot_hyperparam_vs_iteration(self, title=f"{self.kernel_name} surrogate", show=show)
            else:
                raise NameError("Must run active_train before plotting gp_hyperparam.")

        # GP training time vs iteration
        if "gp_timing" in plots:
            if hasattr(self, "training_results"):
                print("Plotting gp timing...")
                vis.plot_train_time_vs_iteration(self, title=f"{self.kernel_name} surrogate", show=show)
            else:
                raise NameError("Must run active_train before plotting gp_timing.")

        # N-D scatterplots and histograms colored by function value
        if "gp_train_corner" in plots:  
            if hasattr(self, "_theta") and hasattr(self, "_y"):
                print("Plotting training sample corner plot...")
                vis.plot_corner_lnp(self, show=show)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        # N-D scatterplots and histograms
        if "gp_train_scatter" in plots:  
            if hasattr(self, "_theta") and hasattr(self, "_y"):
                print("Plotting training sample corner plot...")
                vis.plot_corner_scatter(self, show=show)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        if "gp_train_iteration" in plots:  
            if hasattr(self, "_theta") and hasattr(self, "_y"):
                print("Plotting training sample corner plot...")
                vis.plot_train_sample_vs_iteration(self, show=show)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_iteration.")

        # GP training time vs iteration
        if "gp_fit_2D" in plots:
            if hasattr(self, "_theta") and hasattr(self, "_y"):
                print("Plotting gp fit 2D...")
                if self.ndim == 2:
                    vis.plot_gp_fit_2D(self, ngrid=60, title=f"{self.kernel_name} surrogate", show=show)
                else:
                    print("theta must be 2D to use gp_fit_2D!")
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_fit_2D.")

        # Objective function contour plot
        if "obj_fn_2D" in plots:
            if hasattr(self, "_theta") and hasattr(self, "_y") and hasattr(self, "gp"):
                print("Plotting objective function contours 2D...")
                vis.plot_utility_2D(self, ngrid=60, show=show)
            else:
                raise NameError("Must run init_train and init_gp before plotting obj_fn_2D.")
            
        if "true_fn_2D" in plots:
            if self.ndim == 2:
                print("Plotting true function contours 2D...")
                vis.plot_true_fit_2D(self, ngrid=60, show=show)
            else:
                raise print("theta must be 2D to use true_fn_2D!")

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
                vis.plot_corner(self, self.emcee_samples, sampler="emcee_", show=show)
            else:
                raise NameError("Must run run_emcee before plotting emcee_corner.")

        # emcee walkers
        if "emcee_walkers" in plots:  
            if hasattr(self, "emcee_samples"):
                print("Plotting emcee walkers...")
                vis.plot_emcee_walkers(self, show=show)
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
                vis.plot_corner(self, self.dynesty_samples, sampler="dynesty_", show=show)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_corner.")

        if "dynesty_corner_kde" in plots:  
            if hasattr(self, "dynesty_samples"):
                print("Plotting dynesty posterior kde...")
                vis.plot_corner_kde(self, show=show)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_corner.")

        if "dynesty_traceplot" in plots:
            if hasattr(self, "res"):
                print("Plotting dynesty traceplot...")
                vis.plot_dynesty_traceplot(self, show=show)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_traceplot.")

        if "dynesty_runplot" in plots:
            if hasattr(self, "res"):
                print("Plotting dynesty runplot...")
                vis.plot_dynesty_runplot(self, show=show)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_runplot.")

        # ================================
        # MCMC comparison plots
        # ================================

        if "mcmc_comparison" in plots:
            if hasattr(self, "emcee_samples") and hasattr(self, "res"):
                print("Plotting emcee vs dynesty posterior comparison...")
                vis.plot_mcmc_comparison(self, show=show)
            else:
                raise NameError("Must run run_emcee and run_dynesty before plotting emcee_comparison.")
