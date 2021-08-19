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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    """

    def __init__(self, fn=None, bounds=None, labels=None, prior_sampler=None,
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

        if prior_sampler is None:
            self.prior_sampler = partial(ut.prior_sampler, bounds=self.bounds, sampler='sobol')
        else:
            self.prior_sampler = prior_sampler

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
        """
        Pickle ``SurrogateModel`` object and write summary to a text file
        """

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


    def init_train(self, nsample=None):
        """
        :param nsample: (*int, optional*) 
            Number of samples. Defaults to ``nsample = 50 * self.ndim``

        :param sampler: (*str, optional*) 
            Sampling method. Defaults to ``'sobol'``. 
            See ``utility.prior_sampler`` for more details.
        """

        if nsample is None:
            nsample = 50 * self.ndim

        self.theta0 = self.prior_sampler(nsample=nsample)
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


    def load_test(self, cache_file):

        print(f"Loading test sample from {cache_file}...")
        sims = np.load(cache_file)
        self.theta_test = sims["theta"]
        self.y_test = sims["y"]


    def init_samples(self, train_file=None, test_file=None, reload=False,
                     ntrain=None, ntest=None, sampler=None, scale=True):
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
                    self.init_train(nsample=ntrain)
            else:
                self.init_train(nsample=ntrain)

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

        # record number of training samples
        self.ninit_train = len(self.theta0)
        self.ntrain = self.ninit_train
        self.nactive = 0

        # record number of test samples
        self.ntest = len(self.theta_test)

        self.scale = scale
        if self.scale == True:
            # Create scaling function using the training sample
            self.scaler_t = MinMaxScaler()
            self.scaler_t.fit(np.array(self.bounds).T)

            self.scaler_y = StandardScaler()
            yT = self.y.reshape(1,-1).T
            self.scaler_y.fit(yT)

        
    def init_gp(self, kernel=None, fit_amp=True, fit_mean=True, 
                fit_white_noise=False, white_noise=None, 
                gp_hyper_prior=None, overwrite=False):
        """
        Initialize the Gaussian Process with specified kernel.

        :param kernel: (*str/george kernel obj, optional*) 
            ``george`` kernel object. Defaults to "ExpSquaredKernel". 
            See https://george.readthedocs.io/en/latest/user/kernels/ for more details.
            Options:
                ``'Kernel'``,
                ``'Sum'``,
                ``'Product'``,
                ``'ConstantKernel'``,
                ``'CosineKernel'``,
                ``'DotProductKernel'``,
                ``'EmptyKernel'``,
                ``'ExpKernel'``,
                ``'ExpSine2Kernel'``,
                ``'ExpSquaredKernel'``,
                ``'LinearKernel'``,
                ``'LocalGaussianKernel'``,
                ``'Matern32Kernel'``,
                ``'Matern52Kernel'``,
                ``'MyLocalGaussianKernel'``,
                ``'PolynomialKernel'``,
                ``'RationalQuadraticKernel'``
        """
        
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

        # save white noise to obj
        self.white_noise = white_noise
        self.fit_white_noise = fit_white_noise

        # save amplitude and mean hyperparameter choices
        self.fit_amp = fit_amp
        self.fit_mean = fit_mean
        
        warnings.simplefilter("ignore")
        self.gp = gp_utils.fit_gp(self.theta, self.y, self.kernel, 
                                  fit_amp=self.fit_amp, fit_mean=self.fit_mean,
                                  fit_white_noise=self.fit_white_noise,
                                  white_noise=self.white_noise)

        t0 = time.time()
        op_gp = None
        while op_gp is None:
            op_gp = gp_utils.optimize_gp(self.gp, self.theta, self.y,
                                         gp_hyper_prior=self.gp_hyper_prior)
        self.gp = op_gp
        tf = time.time()

        if self.verbose:
            print(f"optimized hyperparameters: ({np.round(tf - t0, 1)}s)")
            print(self.gp.get_parameter_names())
            print(self.gp.get_parameter_vector())
            print('')


    def evaluate(self, theta_xs):
        """
        Evaluate predictive mean of the GP at point ``theta_xs``

        :param theta_xs: (*array, required*) Point to evaluate GP mean at.

        :returns ypred: (*float*) GP mean evaluated at ``theta_xs``.
        """

        theta_xs = np.asarray(theta_xs).reshape(1,-1)

        # apply the GP
        ypred = self.gp.predict(self.y, theta_xs, return_cov=False)[0]

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
            p0 = self.prior_sampler(nsample=1)
            t0 = minimize(self.neg_evaluate, p0, bounds=tuple(self.bounds), method="nelder-mead")["x"]
        else:
            t0 = None

        thetaN, _ = ut.minimize_objective(self.utility, self.y, self.gp,
                                            bounds=self.bounds,
                                            nopt=nopt,
                                            t0=t0,
                                            ps=self.prior_sampler,
                                            args=(self.y, self.gp, self.bounds))

        # evaluate function at the optimized theta
        yN = self.fn(thetaN)

        return thetaN, yN


    def active_train(self, niter=100, algorithm="bape", gp_opt_freq=10, save_progress=True,
                     opt_init=False): 
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

        # start timing active learning 
        train_t0 = time.time()

        for ii in tqdm.tqdm(range(1, niter+1)):

            # AGP on even, BAPE on odd
            if self.algorithm == "alternate":
                if ii % 2 == 0:
                    self.utility = ut.agp_utility
                else:
                    self.utility = ut.bape_utility

            gp = None
            while gp is None:
                # Find next training point!
                opt_obj_t0 = time.time()
                thetaN, yN = self.find_next_point(opt_init=opt_init)
                opt_obj_tf = time.time()
                
                # add theta and y to training sample
                theta_prop = np.append(self.theta, [thetaN], axis=0)
                y_prop = np.append(self.y, yN)

                fit_gp_t0 = time.time()
                # Fit GP. Make sure to feed in previous iteration hyperparameters!
                gp = gp_utils.fit_gp(theta_prop, y_prop, self.kernel,
                                        hyperparameters=self.gp.get_parameter_vector(),
                                        fit_amp=self.fit_amp, fit_mean=self.fit_mean,
                                        fit_white_noise=self.fit_white_noise,
                                        white_noise=self.white_noise)
                fit_gp_tf = time.time()

            # If proposed (theta, y) did not cause fitting issues, save to surrogate model obj
            self.theta = theta_prop
            self.y = y_prop

            self.gp = gp

            print("Total training samples:", len(self.theta))

            # record active learning train runtime
            self.train_runtime = time.time() - train_t0 

            # evaluate gp training error (scaled)
            ypred = self.gp.predict(self.y, self.theta, return_cov=False, return_var=False)
            if self.scale == True:
                ypred_ = self.scaler_y.transform(ypred.reshape(1,-1).T).flatten()
                y_ = self.scaler_y.transform(self.y.reshape(1,-1).T).flatten()
                training_error = np.mean((y_ - ypred_)**2)
            else:
                training_error = np.mean((y - pred)**2)

            # evaluate gp test error (scaled)
            if hasattr(self, 'theta_test') and hasattr(self, 'y_test'):
                ytest = self.gp.predict(self.y, self.theta_test, return_cov=False, return_var=False)
                if self.scale == True:
                    ytest_ = self.scaler_y.transform(ytest.reshape(1,-1).T).flatten()
                    y_test_ = self.scaler_y.transform(self.y_test.reshape(1,-1).T).flatten()
                    test_error = np.mean((y_test_ - ytest_)**2)
                else:
                    test_error = np.mean((y_test - ytest)**2)
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

            # Optimize GP?
            if (ii + first_iter) % self.gp_opt_freq == 0:
                
                t0 = time.time()
                op_gp = None
                # on first attempt, initialize gp optimization at current hyperparameters
                p0 = self.gp.get_parameter_vector()
                while op_gp is None:
                    op_gp = gp_utils.optimize_gp(gp, self.theta, self.y,
                                            gp_hyper_prior=self.gp_hyper_prior,
                                            p0=p0)
                    # if this initialization doesn't work, try random initialization
                    p0 = None

                self.gp = op_gp
                tf = time.time()

                if self.verbose:
                    print(f"optimized hyperparameters: ({np.round(tf - t0, 1)}s)")
                    print(self.gp.get_parameter_vector())
                
                if (save_progress == True) and (ii != 0):
                    self.save()
                    self.plot(plots=["gp_error", "gp_hyperparam"])
                    if self.ndim == 2:
                        self.plot(plots=["gp_fit_2D"])
                    else:
                        self.plot(plots=["gp_train_scatter"])

        if self.cache:
            self.save()


    def train_gp_default(self):

        sm.init_train()
        sm.init_test()
        sm.init_gp()
        sm.active_train()


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

        theta = np.asarray(theta).reshape(1,-1)

        lnp = self.evaluate(theta) + self.lnprior(theta)

        return lnp


    def run_emcee(self, lnprior=None, nwalkers=None, nsteps=int(5e4), sampler_kwargs={}, run_kwargs={},
                  opt_init=True, multi_proc=True, lnprior_comment=None):
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

    
    def run_dynesty(self, ptform=None, mode='dynamic', sampler_kwargs={}, run_kwargs={},
                    multi_proc=False, ptform_comment=None):
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
            dsampler = DynamicNestedSampler(self.evaluate, 
                                            self.ptform, 
                                            self.ndim,
                                            pool=pool,
                                            **sampler_kwargs)
            print("Initialized dynesty DynamicNestedSampler.")
        elif mode == 'static':
            dsampler = NestedSampler(self.evaluate, 
                                     self.ptform, 
                                     self.ndim,
                                     pool=pool,
                                     **sampler_kwargs)
            print("Initialized dynesty NestedSampler.")
        else:
            raise ValueError(f"mode {mode} is not a valid option. Choose 'dynamic' or 'static'.")


        dsampler.run_nested(**run_kwargs)
        self.res = dsampler.results

        # record dynesty runtime
        self.dynesty_runtime = time.time() - dynesty_t0

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


    def find_map(self, theta0=None, lnprior=None, method="nelder-mead", nRestarts=15, options=None):

        raise NotImplementedError("Not implemented.")


    def plot(self, plots=None, save=True):
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
            gp_plots = ["gp_error", "gp_hyperparam", "gp_timing", "gp_train_corner", "gp_train_scatter"]
            if self.ndim == 2:
                gp_plots.append("gp_fit_2D")
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
                vis.plot_corner_lnp(self)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        # N-D scatterplots and histograms
        if "gp_train_scatter" in plots:  
            if hasattr(self, "theta") and hasattr(self, "y"):
                print("Plotting training sample corner plot...")
                vis.plot_corner_scatter(self)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        if "gp_train_iteration" in plots:  
            if hasattr(self, "theta") and hasattr(self, "y"):
                print("Plotting training sample corner plot...")
                vis.plot_train_sample_vs_iteration(self)
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_iteration.")

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

        # Objective function contour plot
        if "obj_fn_2D" in plots:
            if hasattr(self, "theta") and hasattr(self, "y") and hasattr(self, "gp"):
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
                    