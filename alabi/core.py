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

        # Unit cube scaled bounds
        self.bounds_ = [(0,1) for i in range(self.ndim)]

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


    def init_train(self, nsample=None, sampler='sobol'):
        """
        :param nsample: (*int, optional*) 
            Number of samples. Defaults to ``nsample = 50 * self.ndim``

        :param sampler: (*str, optional*) 
            Sampling method. Defaults to ``'sobol'``. 
            See ``utility.prior_sampler`` for more details.
        """

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
        """
        :param nsample: (*int, optional*) 
            Number of samples. Defaults to ``nsample = 50 * self.ndim``

        :param sampler: (*str, optional*) 
            Sampling method. Defaults to ``'sobol'``. 
            See ``utility.prior_sampler`` for more details.
        """

        if nsample is None:
            nsample = 50 * self.ndim

        self.theta_test = ut.prior_sampler(nsample=nsample, bounds=self.bounds, sampler=sampler)
        self.y_test = ut.eval_fn(self.fn, self.theta_test, ncore=self.ncore)

        # record number of test samples
        self.ntest = len(self.theta_test)

        if self.cache:
            np.savez(f"{self.savedir}/initial_test_sample.npz", theta=self.theta_test, y=self.y_test)


    def scale_data(self, theta, y):

        theta_ = self.scaler_t.transform(theta)

        yT = y.reshape(1,-1).T
        y_ = self.scaler_y.transform(yT).flatten()

        return theta_, y_


    def unscale_data(self, theta_, y_):

        theta = self.scaler_t.inverse_transform(theta_)

        yT_ = y_.reshape(1,-1).T
        y = self.scaler_y.inverse_transform(yT_).flatten()

        return theta, y


    def init_samples(self, train_file=None, test_file=None,
                     ntrain=None, ntest=None, sampler=None):
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

        :param ntrain: (*int, optional*)
            Number of training samples to compute.

        :param ntest: (*int, optional*)
            Number of test samples to compute.
        """

        if train_file is not None:
            sims = np.load(f"{self.savedir}/{train_file}")
            self.theta0 = sims["theta"]
            self.y0 = sims["y"]
            self.theta = self.theta0
            self.y = self.y0

            # record number of training samples
            self.ninit_train = len(self.theta0)
            self.ntrain = self.ninit_train
            self.nactive = 0
        else:
            self.init_train(nsample=ntrain)

        if test_file is not None:
            sims = np.load(f"{self.savedir}/{test_file}")
            self.theta_test = sims["theta"]
            self.y_test = sims["y"]

            # record number of test samples
            self.ntest = len(self.theta_test)
        else:
            self.init_test(nsample=ntest)

        # Create scaling function using the training sample
        self.scaler_t = MinMaxScaler()
        self.scaler_t.fit(np.array(self.bounds).T)

        self.scaler_y = StandardScaler()
        yT = self.y.reshape(1,-1).T
        self.scaler_y.fit(yT)

        # Save the scaled training data to arrays
        self.theta_, self.y_ = self.scale_data(self.theta, self.y)

        # Save the scaled test data to arrays
        self.theta_test_, self.y_test_ = self.scale_data(self.theta_test, self.y_test)

        
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
        
        warnings.simplefilter("ignore")
        self.gp = gp_utils.fit_gp(self.theta_, self.y_, self.kernel, 
                                  fit_amp=fit_amp, fit_mean=fit_mean,
                                  fit_white_noise=fit_white_noise,
                                  white_noise=self.white_noise)

        t0 = time.time()
        self.gp = gp_utils.optimize_gp(self.gp, self.theta_, self.y_,
                                       gp_hyper_prior=self.gp_hyper_prior)
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

        # reshape data array
        theta_xs = np.asarray(theta_xs).reshape(1,-1)

        # scale the data
        theta_xs_ = self.scaler_t.transform(theta_xs)

        # apply the GP
        ypred_ = self.gp.predict(self.y_, theta_xs_, return_cov=False)[0]

        # inverse scale the output
        ypred = self.scaler_y.inverse_transform(ypred_.reshape(1,-1)).flatten()[0]

        return ypred


    def find_next_point(self, nopt=3):
        """
        Find next set of ``(theta, y)`` training points by maximizing the
        active learning utility function.

        :param nopt: (*int, optional*) 
            Number of times to restart the objective function optimization. 
            Defaults to 1. Increase to avoid converging to local maxima.
        """

        thetaN_, _ = ut.minimize_objective(self.utility, self.y_, self.gp,
                                            bounds=self.bounds_,
                                            nopt=nopt,
                                            # t0=self.theta_[np.argmax(self.y_)],
                                            t0=None,
                                            args=(self.y_, self.gp, self.bounds_))

        # unscale theta
        thetaN = self.scaler_t.inverse_transform(thetaN_.reshape(1,-1)).flatten()

        # evaluate function at the optimized theta
        yN = self.fn(thetaN)

        # scale y
        yN_ = self.scaler_y.transform(yN.reshape(1,-1)).flatten()

        return thetaN_, yN_, thetaN, yN


    def active_train(self, niter=100, algorithm="bape", gp_opt_freq=10): 
        """
        :param niter: (*int, optional*)

        :param algorithm: (*str, optional*) 

        :param gp_opt_freq: (*int, optional*)
        """

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

        # start timing active learning 
        train_t0 = time.time()

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
                thetaN_, yN_, thetaN, yN = self.find_next_point()
                opt_obj_tf = time.time()

                # add theta and y to training sample
                theta_prop_ = np.append(self.theta_, [thetaN_], axis=0)
                y_prop_ = np.append(self.y_, yN_)

                theta_prop = np.append(self.theta, [thetaN], axis=0)
                y_prop = np.append(self.y, yN)

                try:
                    fit_gp_t0 = time.time()
                    # Fit GP. Make sure to feed in previous iteration hyperparameters!
                    gp = gp_utils.fit_gp(theta_prop_, y_prop_, self.kernel,
                                         hyperparameters=self.gp.get_parameter_vector())
                    fit_gp_tf = time.time()
                    break

                except:
                    if self.verbose:
                        msg = "Warning: GP fit failed. Likely covariance matrix was not positive definite. "
                        msg += "Attempting another training sample... "
                        msg += "If this issue persists, try adjusting the white_noise or expanding the hyperparameter prior"
                        print(msg)

            # If proposed (theta, y) did not cause fitting issues, save to surrogate model obj
            self.theta_ = theta_prop_
            self.y_ = y_prop_

            self.theta = theta_prop
            self.y = y_prop

            self.gp = gp

            # record active learning train runtime
            self.train_runtime = time.time() - train_t0 

            # evaluate gp training error (scaled)
            ypred_ = self.gp.predict(self.y_, self.theta_, return_cov=False, return_var=False)
            training_error = np.mean((self.y_ - ypred_)**2)

            # evaluate gp test error (scaled)
            if hasattr(self, 'theta_test') and hasattr(self, 'y_test'):
                ytest_ = self.gp.predict(self.y_, self.theta_test_, return_cov=False, return_var=False)
                test_error = np.mean((self.y_test_ - ytest_)**2)
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

            # Optimize GP?
            if ii % self.gp_opt_freq == 0:
                t0 = time.time()
                gp = gp_utils.optimize_gp(gp, self.theta_, self.y_,
                                          gp_hyper_prior=self.gp_hyper_prior)
                tf = time.time()
                if self.verbose:
                    print(f"optimized hyperparameters: ({np.round(tf - t0, 1)}s)")
                    print(self.gp.get_parameter_vector())

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

        return self.evaluate(theta) + self.lnprior(theta)


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
        sampler = emcee.EnsembleSampler(self.nwalkers, 
                                        self.ndim, 
                                        self.lnprob, 
                                        pool=pool,
                                        **sampler_kwargs)

        sampler.run_mcmc(p0, self.nsteps, progress=True, **run_kwargs)

        # record emcee runtime
        self.emcee_runtime = time.time() - emcee_t0

        # burn, thin, and flatten samples
        self.iburn, self.ithin = mcmc_utils.estimateBurnin(sampler, verbose=self.verbose)
        self.emcee_samples_full = sampler.get_chain()
        self.emcee_samples = sampler.get_chain(discard=self.iburn, flat=True, thin=self.ithin) 

        # get acceptance fraction and autocorrelation time
        self.acc_frac = np.mean(sampler.acceptance_fraction)
        self.autcorr_time = np.mean(sampler.get_autocorr_time())
        if self.verbose:
            print(f"Total samples: {self.emcee_samples.shape[0]}")
            print("Mean acceptance fraction: {0:.3f}".format(self.acc_frac))
            print("Mean autocorrelation time: {0:.3f} steps".format(self.autcorr_time))

        # record that emcee has been run
        self.emcee_run = True

        # close pool
        pool.close()

        if self.cache:
            self.save()

            np.savez(f"{self.savedir}/emcee_samples_final.npz", samples=self.emcee_samples)

    
    def run_dynesty(self, ptform=None, sampler_kwargs={}, run_kwargs={},
                     multi_proc=True, ptform_comment=None):
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
        dsampler = DynamicNestedSampler(self.evaluate, 
                                        self.ptform, 
                                        self.ndim,
                                        pool=pool,
                                        **sampler_kwargs)

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
                    