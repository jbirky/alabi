"""
:py:mod:`core.py` 
-------------------------------------
"""

from alabi import utility as ut
from alabi import visualization as vis
from alabi import gp_utils
from alabi import mcmc_utils 
from alabi import cache_utils
from alabi import parallel_utils

import numpy as np
from functools import partial
import george
from george import kernels
import multiprocess as mp
import time
import os
import warnings
import tqdm
import pickle
import scipy.optimize as op

__all__ = ["SurrogateModel", "CachedSurrogateLikelihood"]


class CachedSurrogateLikelihood:
    """
    A picklable cached surrogate likelihood function.
    
    This class creates a callable object that caches the GP computation
    and can be pickled for use with multiprocessing.
    """
    
    def __init__(self, gp_iter, _y_cond, theta_scaler, y_scaler, ndim, return_var=False):
        """
        Initialize the cached surrogate likelihood.
        
        :param gp_iter: Pre-computed GP object
        :param y_cond: Training target values
        :param theta_scaler: Parameter scaler object
        :param y_scaler: Target scaler object  
        :param ndim: Number of dimensions
        """
        self.gp_iter = gp_iter
        self._y_cond = _y_cond
        self.theta_scaler = theta_scaler
        self.y_scaler = y_scaler
        self.ndim = ndim
        self.return_var = return_var
    
    def __call__(self, theta_xs):
        """
        Evaluate the cached surrogate likelihood.
        
        :param theta_xs: Point(s) to evaluate. Same format as surrogate_log_likelihood.
        :param return_var: Whether to return variance as well.
        :returns: Same format as surrogate_log_likelihood.
        """
        # Convert input to numpy array and handle dimensionality
        theta_xs = np.asarray(theta_xs)
        original_shape_1d = False
        
        # Handle 1D input (single point)
        if theta_xs.ndim == 1:
            theta_xs = theta_xs.reshape(1, -1)
            original_shape_1d = True
        elif theta_xs.ndim != 2:
            raise ValueError(f"theta_xs must be 1D or 2D array, got {theta_xs.ndim}D")
        
        # Apply scaling transformation
        _theta_xs = self.theta_scaler.transform(theta_xs)
        
        # Ensure proper shape for george GP
        _theta_xs = np.atleast_2d(_theta_xs)
        if _theta_xs.shape[0] == 1 and _theta_xs.shape[1] != self.ndim:
            if _theta_xs.shape[1] == 1 and _theta_xs.shape[0] == self.ndim:
                _theta_xs = _theta_xs.T
            elif len(_theta_xs.flatten()) == self.ndim:
                _theta_xs = _theta_xs.reshape(1, -1)
        
        # Use the pre-computed GP (this is fast since gp.compute() was already called)
        if self.return_var == False:
            _ypred = self.gp_iter.predict(self._y_cond, _theta_xs, return_var=False, return_cov=False)
            ypred = self.y_scaler.inverse_transform(_ypred.reshape(-1, 1)).flatten()
            
            # Return single value if input was 1D, otherwise return array
            if original_shape_1d:
                return ypred[0]
            else:
                return ypred
                
        else:
            _ypred, _varpred = self.gp_iter.predict(self._y_cond, _theta_xs, return_var=True, return_cov=False)
            ypred = self.y_scaler.inverse_transform(_ypred.reshape(-1, 1)).flatten()
            
            # Variance transformation: variance scales as scale_factor² for linear transforms
            # For FunctionTransformer (like nlog_scaler), we need to handle this carefully
            if hasattr(self.y_scaler, 'scale_') and self.y_scaler.scale_ is not None:
                # For StandardScaler or similar: var_unscaled = scale_factor² × var_scaled  
                var_scale_factor = self.y_scaler.scale_[0] ** 2
                varpred = _varpred * var_scale_factor
            else:
                # For FunctionTransformer or other scalers, compute numerical derivative
                # This is more accurate than using inverse_transform on variance
                try:
                    # Use a small epsilon to estimate the derivative
                    eps = 1e-6
                    test_vals = np.array([[0.0], [eps]])
                    transformed = self.y_scaler.inverse_transform(test_vals)
                    scale_factor = (transformed[1] - transformed[0]) / eps
                    varpred = _varpred * (scale_factor ** 2)
                except:
                    # Fallback: use absolute value of incorrect transform to ensure positivity
                    varpred = np.abs(self.y_scaler.inverse_transform(_varpred.reshape(-1, 1)).flatten())

            # Return single values if input was 1D, otherwise return arrays
            if original_shape_1d:
                return ypred[0], varpred[0]
            else:
                return ypred, varpred
            

class SurrogateModel(object):

    """
    Gaussian Process surrogate model for Bayesian inference and optimization.
    
    A SurrogateModel uses a Gaussian Process to create a fast approximation of expensive
    likelihood functions, enabling efficient Bayesian inference, parameter estimation,
    and active learning. The model supports various active learning algorithms and 
    scalers for handling different types of likelihood functions.

    :param lnlike_fn: (*callable, required*)
        Log-likelihood function that takes parameter array theta and returns scalar 
        log-likelihood value. For Bayesian inference, this is your model's log-likelihood.
        Signature: lnlike_fn(theta) -> float
        
    :param bounds: (*array-like, required*)
        Prior bounds for each parameter. List/array of (min, max) tuples for each dimension.
        Example: bounds = [(0, 1), (2, 3), (-1, 1)]
        
    :param param_names: (*array-like, optional*)
        Names/labels for each parameter. If None, defaults to θ₀, θ₁, etc.
        Length must match number of dimensions in bounds.
        
    :param cache: (*bool, optional, default=True*)
        Whether to cache the trained model to disk for reuse
        
    :param savedir: (*str, optional, default="results/"*)
        Directory for saving results, plots, and cached models
        
    :param model_name: (*str, optional, default="surrogate_model"*)
        Name prefix for cached model files
        
    :param verbose: (*bool, optional, default=True*)
        Print progress information during training and inference
        
    :param ncore: (*int, optional, default=cpu_count()*)
        Number of CPU cores to use for parallel computation
        
    :param ignore_warnings: (*bool, optional, default=True*)
        Suppress sklearn and other package warnings
        
    :param random_state: (*int, optional, default=None*)
        Random seed for reproducible results

    .. attribute:: gp
        :type: george.GP
        
        Trained Gaussian Process model
        
    .. attribute:: bounds
        :type: ndarray
        
        Original parameter bounds (unscaled)
        
    .. attribute:: _bounds
        :type: ndarray
        
        Scaled parameter bounds used for GP training
        
    .. attribute:: _theta
        :type: ndarray
        
        Training parameter samples (scaled)
        
    .. attribute:: _y
        :type: ndarray
        
        Training likelihood values (scaled)
        
    .. attribute:: ntrain
        :type: int
        
        Number of initial training samples
        
    .. attribute:: ndim
        :type: int
        
        Number of parameters/dimensions
        
    .. attribute:: emcee_samples
        :type: ndarray
        
        MCMC samples from emcee (if run_emcee called)
        
    .. attribute:: dynesty_samples
        :type: ndarray
        
        Nested sampling results from dynesty (if run_dynesty called)

    **Examples**
    
    Basic usage for Bayesian inference:
    
    .. code-block:: python
    
        def log_likelihood(theta):
            # Your model likelihood function
            return -0.5 * np.sum((theta - 2)**2)
         
        bounds = [(0, 4), (0, 4)]  # 2D parameter space
        sm = SurrogateModel(log_likelihood, bounds)
        sm.init_samples(ntrain=100)  # Initial training data
        sm.init_gp()  # Initialize Gaussian Process
        sm.active_train(niter=50)  # Active learning
        sm.run_dynesty()  # Bayesian inference
    
    For optimization problems:
    
    .. code-block:: python
    
        sm.active_train(algorithm="jones")  # Use Jones algorithm for optimization
    
    .. seealso::
    
        :meth:`init_samples` : Initialize training data
        :meth:`init_gp` : Initialize Gaussian Process
        :meth:`active_train` : Perform active learning
        :meth:`run_dynesty` : Run nested sampling with dynesty
        :meth:`run_emcee` : Run MCMC sampling with emcee
        :meth:`run_ultranest` : Run nested sampling with UltraNest
        :meth:`run_pymultinest` : Run nested sampling with PyMultiNest
    """

    def __init__(self, lnlike_fn=None, bounds=None, param_names=None, 
                 cache=True, savedir="results/", model_name="surrogate_model",
                 verbose=True, ncore=1, pool_method="forkserver", ignore_warnings=True,
                 random_state=None):

        # Check all required inputs are specified
        if lnlike_fn is None:
            raise ValueError("Must supply lnlike_fn to train GP surrogate model.")
        if bounds is None:
            raise ValueError("Must supply prior bounds.")

        # Set random seed for reproducibility
        if random_state is None:
            # Use a time-based seed to avoid clustering when called repeatedly
            import time
            random_state = int(time.time() * 1000000) % (2**32)
        self.random_state = random_state

        # Set function for training the GP, and initial training samples
        # For bayesian inference problem this would be your log likelihood function
        self.lnlike_fn = lnlike_fn
        self.true_log_likelihood = lnlike_fn

        # unscaled bounds for theta
        self.bounds = np.array(bounds)

        # define prior sampler with unscaled bounds 
        self.prior_sampler = partial(ut.prior_sampler, bounds=self.bounds, sampler="uniform", random_state=None)

        # Determine dimensionality 
        self.ndim = len(self.bounds)
        
        # Set parameter names
        if param_names is not None:
            if len(param_names) != len(bounds):
                raise ValueError("Length of param_names must match length of bounds.")
            self.param_names = param_names
            self.labels = param_names
        else:
            self.param_names = [r"$\theta_%s$"%(i) for i in range(self.ndim)]
            self.labels = ["theta_{i}" for i in range(self.ndim)]

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
        self.pool_method = pool_method
        if ncore <= 0:
            self.ncore = 1
        else:
            self.ncore = min(ncore, mp.cpu_count())
            
        # Check if MPI is active
        self.mpi_is_active = parallel_utils.is_mpi_active()
        if self.mpi_is_active:
            from mpi4py.futures import MPIPoolExecutor

        # false if emcee, dynesty, and ultranest have not been run for this object
        self.emcee_run = False
        self.dynesty_run = False
        self.ultranest_run = False
        
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["pool"] = None
        
        # If self.gp is the problematic object, we need to handle it
        if hasattr(self, 'gp') and hasattr(self.gp, '__dict__'):
            # Try to find and remove unpickleable attributes from gp
            gp_dict = self.gp.__dict__.copy()
            cleaned_gp_dict = {}
            
            for key, value in gp_dict.items():
                try:
                    pickle.dumps(value)
                    cleaned_gp_dict[key] = value
                except:
                    print(f"Removing unpickleable attribute from gp: {key}")
        
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
            
    def _get_pool(self, ncore=None):
        if (ncore is None) or (ncore <= 1):
            return None

        if self.mpi_is_active:
            pool = MPIPoolExecutor(max_workers=ncore)
            return pool
        else:
            pool = mp.get_context(self.pool_method).Pool(processes=ncore)
            return pool

    def _close_pool(self, pool):
        if pool is None:
            pool = None
        elif self.mpi_is_active:
            pool.shutdown(wait=True)
            pool = None
        else:
            pool.close()
            pool.join()
            pool = None

    def save(self):
        """
        Pickle ``SurrogateModel`` object and write summary to a text file.
        """

        file = os.path.join(self.savedir, self.model_name)

        # pickle surrogate model object
        print(f"Caching model to {file}...")
        
        # Create a temporary file first, then rename (atomic operation)
        temp_file = file + ".pkl.tmp"
        try:
            with open(temp_file, "wb") as f:        
                pickle.dump(self, f)
            # Atomic rename to prevent corruption
            os.rename(temp_file, file + ".pkl")
        except Exception as e:
            # Clean up temp file if something went wrong
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

        if hasattr(self, "gp"):
            try:
                cache_utils.write_report_gp(self, file)
            except Exception as e:
                print(f"Error writing GP report: {e}")

        if self.emcee_run == True:
            cache_utils.write_report_emcee(self, file)
            
        if self.dynesty_run == True:
            cache_utils.write_report_dynesty(self, file)

            
    def _lnlike_fn(self, _theta):
        """
        Internal function to evaluate the model function ``lnlike_fn`` at scaled theta.
        This is used to avoid scaling the theta in the main function call.
        """
        # Unscale theta
        theta = self.theta_scaler.inverse_transform(_theta).flatten()
        
        # Evaluate function
        y = self.true_log_likelihood(theta)

        # Scale y - ensure y is a numpy array
        y = np.asarray(y).reshape(-1, 1)
        _y = self.y_scaler.transform(y).flatten()

        return _y
            
    def theta(self):
        """
        Return unscaled training theta values
        """
        
        return self.theta_scaler.inverse_transform(self._theta)
    
    def y(self):
        """
        Return unscaled training y values
        """
        
        return self.y_scaler.inverse_transform(self._y.reshape(-1, 1)).flatten()
    
    
    def refit_scalers(self, theta, y, theta_scaler=None, y_scaler=None):
        """
        Refit the theta and y scalers using current training data.
        Useful if training data has changed significantly.
        """
        if theta_scaler is not None:
            self.theta_scaler = theta_scaler

        if y_scaler is not None:
            self.y_scaler = y_scaler

        self.theta_scaler.fit(self.bounds.T)
        _theta = self.theta_scaler.transform(theta)
        
        y_2d = y.reshape(-1, 1)
        _y = self.y_scaler.fit_transform(y_2d).flatten()
        
        if np.any(np.isnan(_theta)):
            raise ValueError("Refitted theta_scaler produced NaN values!")
        if np.any(np.isinf(_theta)):
            raise ValueError("Refitted theta_scaler produced Inf values!")
        if np.any(np.isnan(_y)):
            raise ValueError("Refitted y_scaler produced NaN values!")
        if np.any(np.isinf(_y)):
            raise ValueError("Refitted y_scaler produced Inf values!")
        
        return _theta, _y


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

        # note: initial samples should be drawn uniformly in scaled space
        # if theta_scaler is a non-linear transform, then samples in real space will be non-uniform
        # _theta = self._prior_sampler(nsample=nsample, sampler=sampler, random_state=None)
        # theta = self.theta_scaler.inverse_transform(_theta)
        theta = self.prior_sampler(nsample=nsample, sampler=sampler, random_state=None) 
        
        # create pool for parallel evaluation of likelihood function
        pool = self._get_pool(ncore=self.ncore)  
        
        # evaluate initial samples in parallel or sequential
        if self.ncore > 1:
            results = list(tqdm.tqdm(
                pool.imap(self.true_log_likelihood, theta),
                total=len(theta)
            ))
            y = np.array(results).reshape(-1, 1)
        else:
            y = np.array([self.true_log_likelihood(tt) for tt in theta]).reshape(-1,1)
            
        # close init_train pool
        self._close_pool(pool)
        
        # replace any nan or inf values
        for ii in range(len(y)):
            if np.isnan(y[ii]) or np.isinf(y[ii]):
                ynan = True
                while ynan == True:
                    # resample theta
                    new_theta = self.prior_sampler(nsample=1, sampler="uniform", random_state=None).flatten()
                    y[ii, 0] = float(self.true_log_likelihood(new_theta))
                    theta[ii] = new_theta
                    if not (np.isnan(y[ii]) or np.isinf(y[ii])):
                        ynan = False
        if self.cache:
            np.savez(f"{self.savedir}/{fname}", theta=theta, y=y)

        return theta, y


    def load_train(self, cache_file):
        """
        Reload training samples from cache file and apply scalers.
        
        :param cache_file: (*str, required*) 
            Name of cache file relative to savedir. Must be a .npz file containing 'theta' and 'y' arrays.

        :returns: (*tuple*) 
            Scaled training samples (_theta, _y) after loading from cache.
        """

        sims = np.load(cache_file)
        theta = sims["theta"]
        y = sims["y"]

        if self.ndim != theta.shape[1]:
            raise ValueError(f"Dimension of bounds (n={self.ndim}) does not \
                              match dimension of training theta (n={theta.shape[1]})")

        return theta, y


    def init_samples(self, ntrain=100, ntest=0, sampler="uniform", train_file=None, test_file=None):
        """
        Initialize training and test samples for the surrogate model.
        
        Creates initial dataset by either loading cached samples or computing new ones
        by evaluating the likelihood function at randomly sampled parameter values.
        
        :param ntrain: (*int, optional, default=100*)
            Number of training samples to generate. Used only if not loading cached samples.
            
        :param ntest: (*int, optional, default=0*)
            Number of test samples to generate. Currently unused.
            
        :param sampler: (*str, optional, default="uniform"*)
            Sampling method for generating parameter values. Options:
            
            - "uniform": Uniform sampling within bounds (default)
            - "sobol": Low-discrepancy Sobol sequence sampling
            - "lhs": Latin hypercube sampling
            
        :param train_file: (*str, optional, default="initial_training_sample.npz"*)
            Filename for cached training samples relative to savedir.
            Format: .npz file containing 'theta' and 'y' arrays.
            
        :param test_file: (*str, optional, default="initial_test_sample.npz"*)
            Filename for cached test samples relative to savedir. Currently unused.
        """

        # Load or create training sample
        if train_file is not None:
            # check if file path exists
            if os.path.exists(train_file):
                cache_file = train_file
            # if not, check if it exists in the savedir
            elif os.path.exists(f"{self.savedir}/{train_file}"):
                cache_file = f"{self.savedir}/{train_file}"
            try:
                theta, y = self.load_train(cache_file)
                print(f"Loaded {len(theta)} train samples from {cache_file}.")
            except Exception as e:
                print(f"Unable to reload {cache_file} due to error: {e}. Computing new samples with {self.ncore} cores...")
                theta, y = self.init_train(nsample=ntrain, sampler=sampler, fname=train_file)
        else:
            train_file="initial_train_file_sample.npz"
            theta, y = self.init_train(nsample=ntrain, sampler=sampler, fname=train_file)
            
        # --------------------------------------------------------
        # Load or create test sample
        if ntest > 0:
            if test_file is not None:
                # check if file path exists
                if os.path.exists(test_file):
                    cache_file = test_file
                # if not, check if it exists in the savedir
                elif os.path.exists(f"{self.savedir}/{test_file}"):
                    cache_file = f"{self.savedir}/{test_file}"
                try:
                    theta_test, y_test = self.load_train(cache_file)
                    print(f"Loaded {len(theta_test)} test samples from {cache_file}.")
                except Exception as e:
                    print(f"Unable to reload {cache_file} due to error: {e}. Computing new samples with {self.ncore} cores...")
                    theta_test, y_test = self.init_train(nsample=ntest, sampler=sampler, fname=test_file)
            else:
                test_file="initial_test_sample.npz"
                theta_test, y_test = self.init_train(nsample=ntest, sampler=sampler, fname=test_file)

            # Save test dataset
            self.theta_test = theta_test
            self.y_test = y_test
            self.ntest = len(theta_test)
        
        else:
            self.theta_test = np.empty((0, self.ndim))
            self.y_test = np.empty(0)
            self.ntest = 0

        # Save initial training sample
        self.theta_train = theta
        self.y_train = y
        
        # record number of training samples
        self.ninit_train = len(theta)
        self.ntrain = self.ninit_train
        self.nactive = 0
            

    def set_hyperparam_prior_bounds(self):
        """
        Configure prior bounds for GP hyperparameters based on current training data.
        
        By default ranges for parameters:
            - mean: [mean(y) - std(y), mean(y) + std(y)]
            - amplitude: [0.1, 10]
            - white noise: [white_noise - 3, white_noise + 3]
        """

        # Configure GP hyperparameter prior
        # hp_bounds = self.gp.get_parameter_bounds()
        # pnames = self.gp.get_parameter_names(include_frozen=False)
        
        if self.uniform_scales == True:
            pnames = self.param_names_optimized
        else:
            pnames = self.param_names_full
        hp_bounds = [[None, None] for _ in pnames]
        
        if self.fit_mean:
            mean_y, std_y = np.mean(self._y), np.std(self._y)
            mean_bounds = [mean_y - std_y, mean_y + std_y]
            hp_bounds[pnames.index("mean:value")] = mean_bounds
            
        if self.fit_amp:
            amp_bounds = [np.var(self._y) * 10**self.gp_amp_rng[0], np.var(self._y) * 10**self.gp_amp_rng[1]] 
            hp_bounds[pnames.index(f"{self.kernel_amp_key}:log_constant")] = amp_bounds

        if self.fit_white_noise:
            wn_bounds = [self.white_noise - 3, self.white_noise + 3]
            hp_bounds[pnames.index("white_noise:value")] = wn_bounds
            
        if self.uniform_scales == True:
            hp_bounds[pnames.index(f"{self.kernel_scale_key}:metric:log_M")] = self.gp_scale_rng
        else:
            for ii in range(self.ndim):
                hp_bounds[pnames.index(f"{self.kernel_scale_key}:metric:log_M_{ii}_{ii}")] = self.gp_scale_rng

        self.hp_bounds = np.array(hp_bounds)
        self.gp_hyper_prior = partial(ut.lnprior_uniform, bounds=self.hp_bounds)
        
    
    def expand_hyperparameter_vector(self, optimized_params):
        
        if optimized_params is None:
            raise ValueError("optimized_params cannot be None")
            
        if self.uniform_scales == False:
            return optimized_params
        
        full_params = np.ones(len(self.param_names_full))
        
        if self.fit_mean:
            full_params[self.param_names_full.index("mean:value")] = optimized_params[self.param_names_optimized.index("mean:value")]
        if self.fit_amp:
            full_params[self.param_names_full.index(f"{self.kernel_amp_key}:log_constant")] = optimized_params[self.param_names_optimized.index(f"{self.kernel_amp_key}:log_constant")]
        if self.fit_white_noise:
            full_params[self.param_names_full.index("white_noise:value")] = optimized_params[self.param_names_optimized.index("white_noise:value")]
        
        # if self.uniform_scales == True:  use same scale length for all dimensions
        for ii in range(self.ndim):
            full_params[self.param_names_full.index(f"{self.kernel_scale_key}:metric:log_M_{ii}_{ii}")] = optimized_params[self.param_names_optimized.index(f"{self.kernel_scale_key}:metric:log_M")]
   
        return np.array(full_params)
        
        
    def set_hyperparameter_vector(self, tmp_gp, optimized_params):
        
        if optimized_params is None:
            raise ValueError("optimized_params cannot be None. Cannot set hyperparameters.")
            
        if self.uniform_scales == True:
            full_params = self.expand_hyperparameter_vector(optimized_params)
        else:
            full_params = optimized_params
                
        tmp_gp.set_parameter_vector(full_params)
            
        return tmp_gp
    
    
    def get_hyperparameter_dict(self, gp):
        """
        Get current GP hyperparameters as a dictionary.
        
        :returns: (*dict*) 
            Dictionary of current GP hyperparameters with names as keys and values as values.
        """
        
        hp_dict = gp.get_parameter_dict()
        
        if self.uniform_scales == True:
            hp_dict[f"{self.kernel_scale_key}:metric:log_M"] = hp_dict.pop(f"{self.kernel_scale_key}:metric:log_M_0_0")
            for ii in range(1, self.ndim):
                del hp_dict[f"{self.kernel_scale_key}:metric:log_M_{ii}_{ii}"]
        
        return hp_dict
    
    
    def get_hyperparameter_vector(self, gp):
        
        hp_dict = self.get_hyperparameter_dict(gp)
        hp_vector = np.fromiter(hp_dict.values(), dtype=float)
        
        return hp_vector

        
    def init_gp(self, 
                kernel="ExpSquaredKernel",
                fit_amp=True, 
                fit_mean=True, 
                fit_white_noise=True, 
                white_noise=-12, 
                gp_scale_rng=[-2,2],
                gp_amp_rng=[-1,1],
                uniform_scales=False,
                overwrite=False,
                theta_scaler=ut.no_scaler,
                y_scaler=ut.no_scaler,
                gp_opt_method="l-bfgs-b", 
                gp_nopt=3,
                optimizer_kwargs={"maxiter": 100, "xatol": 1e-4, "fatol": 1e-3, "adaptive": True},
                hyperopt_method="cv",
                regularize=True,
                amp_0=1.0,
                mu_0=1.0,
                sigma_0=2.0,
                cv_folds=5,
                cv_scoring="mse",
                cv_n_candidates=100,
                cv_stage2_candidates=50,
                cv_stage2_width=0.5,
                cv_stage3_candidates=25,
                cv_stage3_width=0.25,
                cv_weighted_factor=1.0,
                multi_proc=True):
        """
        Initialize the Gaussian Process surrogate model with specified kernel and hyperparameters.

        This function sets up a Gaussian Process (GP) using the george library with the specified 
        kernel type and configuration. The GP is initialized with random scale lengths and then
        fitted to the current training data.

        :param kernel: (*str or george kernel object, optional*) 
            Kernel type for the Gaussian Process. Can be either a string specifying one of the 
            built-in kernels or a george kernel object. Default is "ExpSquaredKernel".
            
            Built-in options:
                - ``'ExpSquaredKernel'``: Squared exponential (RBF) kernel, smooth functions
                - ``'Matern32Kernel'``: Matérn kernel with ν=3/2, moderately smooth functions  
                - ``'Matern52Kernel'``: Matérn kernel with ν=5/2, smooth functions
                - ``'RationalQuadraticKernel'``: Rational quadratic kernel, scale mixture of RBF kernels
                
            See https://george.readthedocs.io/en/latest/user/kernels/ for more details.

        :param fit_amp: (*bool, optional*) 
            Whether to optimize the amplitude (overall scale) hyperparameter of the kernel.
            If True, the GP will learn the optimal amplitude from data. Default is True.

        :param fit_mean: (*bool, optional*) 
            Whether to optimize the mean function hyperparameter. If True, the GP will learn
            a constant mean offset. If False, assumes zero mean. Default is True.

        :param fit_white_noise: (*bool, optional*) 
            Whether to optimize the white noise (nugget) hyperparameter. If True, the GP will
            learn the optimal noise level. If False, uses the fixed value from white_noise.
            Default is True.

        :param white_noise: (*float, optional*) 
            Log-scale white noise parameter. If fit_white_noise=False, this fixed value is used.
            If fit_white_noise=True, this serves as the initial guess. Typical values are 
            between -15 (very low noise) and -5 (high noise). Default is -12.

        :param gp_scale_rng: (*list of two floats, optional*) 
            Log-scale bounds for the characteristic length scale parameters of the kernel.
            Format: [log_min_scale, log_max_scale]. These bounds apply to all input dimensions.
            Default is [-2, 2], corresponding to scales between ~0.14 and ~7.4 in original units.
            
        :param uniform_scales: (*bool, optional*) 
            If True, the same scale length will be used for all input dimensions. If False, each dimension
            will have its own independent scale length. Default is False.

        :param overwrite: (*bool, optional*) 
            If True, allows reinitializing the GP even if one already exists. If False and a GP
            already exists, raises an AssertionError. Default is False.
            
        :param theta_scaler: (*sklearn transformer, optional, default=no_scaler*)
            Scaler for input parameters. Applied to theta values before GP training.
            Common options: MinMaxScaler() (scale to [0,1]) or StandardScaler()
            
        :param y_scaler: (*sklearn transformer, optional, default=no_scaler*)
            Scaler for output values (log-likelihoods). Options include:
            - no_scaler: No scaling (default)
            - minmax_scaler: Scale to [0,1] 
            - nlog_scaler: Apply -log10(-y) transformation for negative log-likelihoods
            - log_scaler: Apply log10(y) for positive values

        :param gp_opt_method: (*str, optional*) 
            Optimization method for GP hyperparameter optimization. Passed to scipy.optimize.minimize.
            Common options: 'l-bfgs-b', 'newton-cg', 'bfgs', 'cg'. Default is 'l-bfgs-b'.

        :param gp_nopt: (*int, optional*) 
            Number of optimization restarts for GP hyperparameter optimization. Multiple restarts
            help avoid local minima. Default is 3.

        :param optimizer_kwargs: (*dict, optional*) 
            Additional keyword arguments passed to the scipy optimizer. Common options include
            'maxiter' (maximum iterations) and convergence tolerances. 
            Default is {"maxiter": 50}.

        :param hyperopt_method: (*str, optional, default='ml'*)
            Method for optimizing GP hyperparameters:
            
            - 'ml': Maximum marginal likelihood (fast, may overfit)
            - 'cv': k-fold cross-validation (slower, prevents overfitting)

        :param cv_folds: (*int, optional, default=5*)
            Number of folds for cross-validation (only used if hyperopt_method='cv').

        :param cv_scoring: (*str, optional, default='mse'*)
            Scoring metric for cross-validation. Options: 'mse', 'mae', 'r2'.

        :param cv_n_candidates: (*int, optional, default=20*)
            Number of hyperparameter candidates to evaluate for CV.

        :param cv_stage2_candidates: (*int, optional, default=20*)
            Number of candidates for stage 2 grid search. Only used when cv_two_stage=True.

        :param cv_stage2_width: (*float, optional, default=0.3*)
            Width factor for stage 2 search around best parameters from stage 1.
            Smaller values = tighter search. Only used when cv_two_stage=True.

        :param cv_stage3_candidates: (*int, optional, default=None*)
            Number of candidates for stage 3 ultra-fine search. If None, uses 
            max(cv_stage2_candidates // 2, 3). Only used when cv_three_stage=True.

        :param cv_stage3_width: (*float, optional, default=0.2*)
            Width factor for stage 3 search around best parameters from stage 2.
            Should be smaller than cv_stage2_width for finer refinement.
            Only used when cv_three_stage=True.

        :raises RuntimeError:
            If init_samples() has not been called before init_gp().
        :raises AssertionError:
            If a GP already exists and overwrite=False.
        :raises ValueError:
            If an invalid kernel name is provided.
        :raises Exception:
            If GP initialization fails after multiple attempts with different scale lengths.

        .. note:: 
        
            This function must be called after init_samples() since it requires training data
            to initialize the GP. The function will automatically retry initialization with
            different random scale lengths if the initial attempt fails.

        .. code-block:: python

            >>> # Basic initialization with default settings
            >>> sm.init_gp()
            
            >>> # Custom kernel with specific hyperparameter settings
            >>> sm.init_gp(kernel="Matern52Kernel", 
            ...            fit_white_noise=False, 
            ...            white_noise=-10,
            ...            gp_scale_rng=[-1, 1])
            
            >>> # High-precision optimization
            >>> sm.init_gp(gp_opt_method="l-bfgs-b",
            ...            gp_nopt=5,
            ...            optimizer_kwargs={"maxiter": 100, "ftol": 1e-9})
        """
        
        if not hasattr(self, 'theta_train'):
            raise RuntimeError(
                "No training data found. Call init_samples() before init_gp().")

        if hasattr(self, 'gp') and (overwrite == False):
            raise AssertionError(
                "GP kernel already assigned. Use overwrite=True to re-assign the kernel.")
            
        # optional hyperparameter choices
        self.fit_amp = fit_amp
        self.fit_mean = fit_mean
        if white_noise is None and fit_white_noise:
            raise ValueError(
                "white_noise cannot be None when fit_white_noise=True. "
                "Either set fit_white_noise=False or provide a numeric "
                "white_noise value (default is -12).")
        self.fit_white_noise = fit_white_noise
        self.white_noise = white_noise
        self.uniform_scales = uniform_scales

        # GP hyperparameter optimization method
        self.gp_opt_method = gp_opt_method

        # GP hyperparameter number of opt restarts
        self.gp_nopt = gp_nopt

        # Save all kwargs needed for opt_gp function
        self.opt_gp_kwargs = {"hyperopt_method": hyperopt_method,
                              "regularize": regularize,
                              "amp_0": amp_0,
                              "mu_0": mu_0,
                              "sigma_0": sigma_0,
                              "optimizer_kwargs": optimizer_kwargs,
                              "cv_folds": cv_folds,
                              "cv_scoring": cv_scoring,
                              "cv_n_candidates": cv_n_candidates,
                              "cv_stage2_candidates": cv_stage2_candidates,
                              "cv_stage2_width": cv_stage2_width,
                              "cv_stage3_candidates": cv_stage3_candidates,
                              "cv_stage3_width": cv_stage3_width,
                              "cv_weighted_factor": cv_weighted_factor,
                              "multi_proc": multi_proc,
                              }
        
        # -------------------------------------------------------------------------
        # define scaling functions
        # Scale inputs between 0 and 1
        self.theta_scaler = theta_scaler
        self.theta_scaler.fit(self.bounds.T)
        
        # Scale bounds to [0, 1] for training
        self._bounds = self.theta_scaler.transform(self.bounds.T).T
        self._prior_sampler = partial(ut.prior_sampler, bounds=self._bounds, sampler="uniform", random_state=None)

        # Output scaling function
        self.y_scaler = y_scaler
        
        # Fit scalers
        self._theta, self._y = self.refit_scalers(self.theta_train, self.y_train)
        
        # Save scaled training data for GP fitting
        self._theta_test = self.theta_scaler.transform(self.theta_test)
        self._y_test = self.y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()
        self._theta_train = self._theta
        self._y_train = self._y

        self.training_results = {"iteration" : [], 
                                 "gp_hyperparameters" : [],  
                                 "gp_hyperparameter_opt_iteration" : [],
                                 "gp_hyperparam_opt_time" : [],
                                 "training_mse" : [],
                                 "test_mse" : [], 
                                 "training_scaled_mse" : [],
                                 "test_scaled_mse" : [],
                                 "gp_kl_divergence" : [],
                                 "gp_train_time" : [],
                                 "obj_fn_opt_time" : [],
                                 "acquisition_optimizer_niter" : []
                                 }
        
        # -------------------------------------------------------------------------
        # set the bounds for scale length parameters
        self.gp_scale_rng = gp_scale_rng
        self.gp_amp_rng = gp_amp_rng
        
        # metric_bounds expects log-scale bounds
        log_metric_bounds = [(min(gp_scale_rng), max(gp_scale_rng)) for _ in range(self.ndim)]
        metric_bounds = [(np.e**min(gp_scale_rng), np.e**max(gp_scale_rng)) for _ in range(self.ndim)]

        valid_scales = False
        max_attempts = 10  # Prevent infinite loops
        attempt = 0
        
        while valid_scales == False and attempt < max_attempts:
            attempt += 1
            
            # Generate initial scale length in linear scale (metric parameter expects linear scale)
            # gp_scale_rng is in log scale, so convert to linear scale for initial guess
            log_initial_lscale = np.random.uniform(min(gp_scale_rng), max(gp_scale_rng), self.ndim)
            initial_lscale = np.exp(log_initial_lscale)
            
            # Note: metric is linear scale, but metric_bounds are log scale!
            # https://github.com/dfm/george/issues/150
            
            # Initialize GP kernel
            try:
                # Stationary kernels
                if kernel == "ExpSquaredKernel":
                    # Guess initial metric, or scale length of the covariances (must be > 0)
                    self.kernel = kernels.ExpSquaredKernel(metric=initial_lscale, metric_bounds=log_metric_bounds, ndim=self.ndim)
                    self.kernel_name = kernel
                    print("Initialized GP with squared exponential kernel.")
                elif kernel == "RationalQuadraticKernel":
                    self.kernel = kernels.RationalQuadraticKernel(log_alpha=1, metric=initial_lscale, metric_bounds=log_metric_bounds, ndim=self.ndim)
                    self.kernel_name = kernel
                    print("Initialized GP with rational quadratic kernel.")
                elif kernel == "Matern32Kernel":
                    self.kernel = kernels.Matern32Kernel(metric=initial_lscale, metric_bounds=log_metric_bounds, ndim=self.ndim)
                    self.kernel_name = kernel
                    print("Initialized GP with Matérn-3/2 kernel.")
                elif kernel == "Matern52Kernel":
                    self.kernel = kernels.Matern52Kernel(metric=initial_lscale, metric_bounds=log_metric_bounds, ndim=self.ndim)
                    self.kernel_name = kernel
                    print("Initialized GP with Matérn-5/2 kernel.")
                else:
                    raise ValueError(f"Kernel '{kernel}' is not a valid option. Valid options: ExpSquaredKernel, Matern32Kernel, Matern52Kernel, RationalQuadraticKernel")
                
                # create GP first time 
                self.gp = gp_utils.configure_gp(self._theta, self._y, self.kernel, 
                                                fit_amp=self.fit_amp, 
                                                fit_mean=self.fit_mean,
                                                fit_white_noise=self.fit_white_noise,
                                                white_noise=self.white_noise)
                if self.gp is None:
                    print(f"Warning: configure_gp returned None.")
                    print(f"Data shape: theta={self._theta.shape}, y={self._y.shape}")
                    print(f"Scale lengths: {initial_lscale} (log: {log_initial_lscale})")
                    print(f"Scale bounds: {metric_bounds} (log: {log_metric_bounds})")
                    print(f"Retrying with new initial scale length...\n")
                    continue
                else:
                    valid_scales = True
                    print(f"Successfully initialized GP on attempt {attempt}")
                    
            except Exception as e:
                print(f"Exception during GP initialization on attempt {attempt}:")
                print(f"  Kernel: {kernel}")
                print(f"  Initial scales: {initial_lscale}")
                print(f"  Scale bounds: {log_metric_bounds}")
                print(f"  Data shape: theta={self._theta.shape}, y={self._y.shape}")
                print(f"  Exception: {type(e).__name__}: {e}")
                print("Retrying with new initial scale length...\n")
                continue
        
        if not valid_scales:
            raise RuntimeError(f"Failed to initialize GP after {max_attempts} attempts. "
                               f"Check your data, kernel choice, and scale bounds. "
                               f"Current settings: kernel={kernel}, gp_scale_rng={gp_scale_rng}")
                
        self.param_names_full = self.gp.get_parameter_names(include_frozen=False)
        self.param_names_optimized = []
        
        self.kernel_scale_key = list(filter(lambda x: "metric:log_M" in x, self.param_names_full))[0].split(":metric:log_M")[0]
        
        if fit_mean:
            self.param_names_optimized.append("mean:value")
        if fit_amp:
            self.kernel_amp_key = list(filter(lambda x: "log_constant" in x, self.param_names_full))[0].split(":log_constant")[0]
            self.param_names_optimized.append(f"{self.kernel_amp_key}:log_constant")
        if fit_white_noise:
            self.param_names_optimized.append("white_noise:value")
        if self.uniform_scales:
            self.param_names_optimized.append(f"{self.kernel_scale_key}:metric:log_M")
        else:
            for ii in range(self.ndim):
                self.param_names_optimized.append(f"{self.kernel_scale_key}:metric:log_M_{ii}_{ii}")
                
        # Infer lengthscale indices (used if regularization is enabled)
        self.hp_length_indices = []
        self.hp_other_indices = []
        for ii, name in enumerate(self.param_names_full):
            # Common patterns for lengthscale parameters in george kernels
            if any(pattern in name.lower() for pattern in ["metric:log_m"]):
                self.hp_length_indices.append(ii)
            else:
                self.hp_other_indices.append(ii)
        if self.uniform_scales:
            # Only one lengthscale parameter when uniform scales are used
            self.hp_length_index = [self.param_names_optimized.index(f"{self.kernel_scale_key}:metric:log_M")]
            
        # record initial hyperparameters
        self.initial_gp_hyperparameters = self.get_hyperparameter_vector(self.gp)
        
        # Optimize GP hyperparameters
        self.gp, _ = self._opt_gp(**self.opt_gp_kwargs)
        
        if hasattr(self, "_theta_test") and hasattr(self, "_y_test") and len(self._y_test) > 0:
            _ytest = self.gp.predict(self._y, self._theta_test, return_cov=False, return_var=False)
            ytest = self.y_scaler.inverse_transform(_ytest.reshape(-1, 1)).flatten()
            ytest_true = self.y_scaler.inverse_transform(self._y_test.reshape(-1, 1)).flatten()
            test_mse = np.mean((ytest_true - ytest)**2)
            return test_mse
        else:
            return None
        
        
    def _fit_gp(self, _theta=None, _y=None, hyperparameters=None):
        """
        Fit Gaussian Process to training data with current hyperparameters.
        
        :param _theta: (*array, optional*) 
            Scaled training parameter samples. If None, uses ``self._theta``.
            
        :param _y: (*array, optional*) 
            Scaled training output values. If None, uses ``self._y``.
            
        :returns: (*tuple*) 
            - gp: Fitted george.GP object
            - timing: Time taken to fit the GP in seconds
        """

        if _theta is None:
            _theta = self._theta

        if _y is None:
            _y = self._y

        t0 = time.time()

        self.set_hyperparam_prior_bounds()

        # Validate input data
        if not np.all(np.isfinite(_theta)):
            raise ValueError(f"_theta contains NaN or Inf values")
        if not np.all(np.isfinite(_y)):
            raise ValueError(f"_y contains NaN or Inf values: {_y[~np.isfinite(_y)]}")

        y_median = np.median(_y)
        y_var = np.var(_y)
        
        if not np.isfinite(y_median):
            raise ValueError(f"median(_y) is not finite: {y_median}")
        if not np.isfinite(y_var) or y_var == 0:
            raise ValueError(f"var(_y) is not finite or zero: {y_var}")

        if self.fit_amp == True:
            kernel = self.kernel * y_var
        else:
            kernel = self.kernel

        gp = george.GP(kernel=kernel, fit_mean=self.fit_mean, mean=y_median,
                    white_noise=self.white_noise, fit_white_noise=self.fit_white_noise)

        # Check if hyperparameters contain NaN or Inf
        if hyperparameters is not None:
            hyperparameters_array = np.atleast_1d(hyperparameters)
            if not np.all(np.isfinite(hyperparameters_array)):
                print(f"Warning: Hyperparameters contain NaN or Inf: {hyperparameters_array}")
                print("Reoptimizing hyperparameters from scratch...")
                gp, _ = self._opt_gp(**self.opt_gp_kwargs, _theta=_theta, _y=_y)
                # Validate the reoptimized GP — clip if still invalid
                reopt_params = gp.get_parameter_vector()
                if not np.all(np.isfinite(reopt_params)):
                    print(f"Warning: Reoptimized GP still has invalid parameters: {reopt_params}")
                    reopt_params = np.where(np.isfinite(reopt_params), reopt_params, 0.0)
                    print(f"Clipped to: {reopt_params}")
                    gp.set_parameter_vector(reopt_params)
                    gp.compute(_theta)
                return gp, time.time() - t0
        
        gp = self.set_hyperparameter_vector(gp, hyperparameters)
        gp.compute(_theta)

        return gp, time.time() - t0
    
    
    def _opt_gp(self, hyperopt_method="ml", regularize=True, amp_0=1.0, mu_0=1.0, sigma_0=2.0,
               optimizer_kwargs={"maxiter": 100, "xatol": 1e-4, "fatol": 1e-3, "adaptive": True},
               cv_folds=5, cv_scoring="mse", cv_n_candidates=20, multi_proc=True, 
               cv_stage2_candidates=None, cv_stage2_width=0.5, cv_stage3_candidates=None, cv_stage3_width=0.2,
               cv_weighted_mse_method="exponential", cv_weighted_factor=1.0, _theta=None, _y=None,
               theta_scaler=None, y_scaler=None):
        """
        Optimize GP hyperparameters using marginal likelihood or cross-validation.
        
        :param method: (*str, optional, default="ml"*)
            Optimization method to use:
            
            - "ml": Maximum marginal likelihood (default, fast)
            - "cv": k-fold cross-validation (slower but prevents overfitting)
            
        :param cv_folds: (*int, optional, default=5*)
            Number of folds for cross-validation (only used if method="cv")
            
        :param cv_scoring: (*str, optional, default='mse'*)
            Scoring metric for cross-validation. Options: 'mse', 'mae', 'r2'
            
        :param cv_n_candidates: (*int, optional, default=20*)
            Number of hyperparameter candidates to evaluate for CV
            
        :param multi_proc: (*bool, optional, default=True*)
            Whether to use multiprocessing for CV evaluation.
            Only used when method="cv".
            
        :param cv_two_stage: (*bool, optional, default=False*)
            Whether to use two-stage CV optimization (explore-exploit strategy).
            Only used when method="cv".
            
        :param cv_stage2_candidates: (*int, optional, default=None*)
            Number of candidates for stage 2 grid search. If None, uses
            cv_n_candidates // 2.
            
        :param cv_stage2_width: (*float, optional, default=0.5*)
            Width factor for stage 2 search around best parameters.
            Smaller values = tighter search around best from stage 1.
            
        :param cv_three_stage: (*bool, optional, default=False*)
            Whether to use three-stage CV optimization (explore-exploit-refine).
            Requires cv_two_stage=True. Only used when method="cv".
            
        :param cv_stage3_candidates: (*int, optional, default=None*)
            Number of candidates for stage 3 ultra-fine search. If None, uses
            max(cv_stage2_candidates // 2, 3). Only used when cv_three_stage=True.
            
        :param cv_stage3_width: (*float, optional, default=0.2*)
            Width factor for stage 3 search around stage 2 best parameters.
            Should be smaller than cv_stage2_width for finer refinement.
            
        :returns: (*tuple*) 
            - op_gp: Optimized george.GP object with updated hyperparameters
            - timing: Time taken for optimization in seconds
        """

        t0 = time.time()
        
        # Use provided _theta and _y, or fall back to self._theta and self._y
        if _theta is None:
            _theta = self._theta
        if _y is None:
            _y = self._y
        
        if hyperopt_method.lower() not in ["ml", "cv"]:
            hyperopt_method = "ml"
            print(f"Invalid method '{hyperopt_method}'. Must be 'ml' or 'cv'. Defaulting to 'ml'.")
        
        if self.gp_opt_method in ["newton-cg", "l-bfgs-b"]:
            use_gradient = True

        self.set_hyperparam_prior_bounds()
            
        if hyperopt_method.lower() == "ml":
            current_gp = self.gp
            current_gp.compute(_theta)

            # Define the objective function (negative log-likelihood in this case).
            def nll(p_opt):
                if self.uniform_scales:
                    p = self.expand_hyperparameter_vector(p_opt)
                else:
                    p = p_opt
                tmp_gp = self.set_hyperparameter_vector(current_gp, p)
                ll = -tmp_gp.log_likelihood(_y, quiet=True)
                if regularize:
                    reg = gp_utils.regularization_term(p, self.hp_length_indices, amp_0=amp_0, mu_0=mu_0, sigma_0=sigma_0)
                    ll += reg
                return ll if np.isfinite(ll) else 1e25

            # And the gradient of the objective function.
            def grad_nll(p_opt):
                if self.uniform_scales:
                    p = self.expand_hyperparameter_vector(p_opt)
                else:
                    p = p_opt
                tmp_gp = self.set_hyperparameter_vector(current_gp, p)
                grad_lnlike = -tmp_gp.grad_log_likelihood(_y, quiet=True)
                
                if self.uniform_scales:
                    gll = np.zeros(len(p_opt))
                    gll[self.hp_length_index] = np.mean(grad_lnlike[self.hp_length_indices])
                    gll[self.hp_other_indices] = grad_lnlike[self.hp_other_indices]
                else:
                    gll = grad_lnlike
                        
                if regularize:
                    reg_grad = gp_utils.regularization_gradient(p, self.hp_length_indices, amp_0=amp_0, mu_0=mu_0, sigma_0=sigma_0)           
                    if self.uniform_scales:
                        reg_grad_avg = np.mean(reg_grad[self.hp_length_indices])
                        gll[self.hp_length_index] += reg_grad_avg
                    else:
                        gll += reg_grad

                return gll
            
            if use_gradient:
                jac = grad_nll
            else:
                jac = None
            
            def _optimize_fn(x0):
                return op.minimize(fun=nll, x0=x0, jac=jac, method=self.gp_opt_method, bounds=self.hp_bounds, options=optimizer_kwargs)

            current_hp = self.get_hyperparameter_vector(current_gp)
            
            if self.gp_nopt <= 1:
                results = _optimize_fn(current_hp)
            else:
                p0 = ut.prior_sampler(bounds=self.hp_bounds, nsample=self.gp_nopt, sampler="lhs", random_state=None)
                p0[0] = current_hp
                
                if self.ncore <= 1:
                    # Sequential with progress bar
                    opt_results = [_optimize_fn(p) for p in tqdm.tqdm(p0, desc="Optimizing GP")]
                else:
                    pool = self._get_pool(ncore=self.ncore)
                    # Parallel with progress bar using imap
                    opt_results = list(tqdm.tqdm(
                        pool.imap(_optimize_fn, p0),
                        total=len(p0),
                        desc=f"Optimizing GP ({self.ncore} cores)"
                    ))
                    self._close_pool(pool)

                def get_fun_value(res):
                    return res.fun
                results = min(opt_results, key=get_fun_value)

            # Clip optimized parameters to bounds and validate
            if self.hp_bounds is not None:
                results.x = np.clip(results.x, self.hp_bounds[:, 0], self.hp_bounds[:, 1])
            if not np.all(np.isfinite(results.x)):
                print(f"Warning: GP optimization produced non-finite parameters: {results.x}")
                print("Falling back to current hyperparameters.")
                results.x = current_hp

            op_gp = self.set_hyperparameter_vector(current_gp, results.x)
            op_gp.compute(_theta)
            
            if self.verbose:
                nll_0 = nll(current_hp)
                nll_fit = nll(results.x)
                if regularize:
                    reg_0 = gp_utils.regularization_term(self.expand_hyperparameter_vector(current_hp), self.hp_length_indices, amp_0=amp_0, mu_0=mu_0, sigma_0=sigma_0)
                    reg_fit = gp_utils.regularization_term(self.expand_hyperparameter_vector(results.x), self.hp_length_indices, amp_0=amp_0, mu_0=mu_0, sigma_0=sigma_0)
                    print(f"Initial -logL: \t {nll_0:.4f} | \t Regularization: {reg_0:.4f} | \t Total: {nll_0 + reg_0:.4f}")
                    print(f"Final -logL: \t {nll_fit:.4f} | \t Regularization: {reg_fit:.4f} | \t Total: {nll_fit + reg_fit:.4f} ")
                else:
                    print(f"Initial -logL: \t {nll_0:.4f} | \t Regularization: None")
                    print(f"Final -logL: \t {nll_fit:.4f} | \t Regularization: None ")
                print(f"{results.nit} iterations | Success: {results.success} | Message: {results.message} \n")

        if hyperopt_method.lower() == "cv":
            # Cross-validation hyperparameter optimization    
            if self.verbose:
                print(f"\nOptimizing GP hyperparameters using {cv_folds}-fold cross-validation...")
            
            try:                         
                candidates = ut.prior_sampler(bounds=self.hp_bounds, nsample=cv_n_candidates, sampler="lhs", random_state=None)

                # Add current hyperparameters as a candidate if GP exists
                if hasattr(self, "gp"):
                    candidates[0] = self.get_hyperparameter_vector(self.gp)
                
                # Expand hyperparameters if using uniform scales
                # CV function expects full parameter vectors that can be set directly on GP
                if self.uniform_scales:
                    candidates_expanded = np.array([self.expand_hyperparameter_vector(c) for c in candidates])
                else:
                    candidates_expanded = candidates
                     
                # suppress outputs if running parallel chains   
                if multi_proc:
                    verbose_cv = False  # Suppress for parallel chains
                else:
                    verbose_cv = True  # Always show CV diagnostics
                
                # Optimize using cross-validation
                pool = self._get_pool(ncore=self.ncore) if multi_proc else None
                op_gp = gp_utils.optimize_gp_kfold_cv(
                    self.gp, _theta, _y,
                    candidates_expanded,
                    self.y_scaler,
                    k_folds=cv_folds,
                    scoring=cv_scoring,
                    pool=pool,
                    stage2_candidates=cv_stage2_candidates,
                    stage2_width=cv_stage2_width,
                    stage3_candidates=cv_stage3_candidates,
                    stage3_width=cv_stage3_width,
                    weighted_mse_method=cv_weighted_mse_method,
                    weighted_mse_factor=cv_weighted_factor,
                    verbose=verbose_cv
                )
                self._close_pool(pool)
                
            except Exception as e:
                import traceback
                import sys
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_line = traceback.extract_tb(exc_traceback)[-1]
                print(f"Warning: CV hyperparameter optimization failed: {str(e)}")
                print(f"Error at line {tb_line.lineno} in {tb_line.filename}: {tb_line.line}")
                print("Falling back to maximum likelihood optimization...")
                
                # Fall back to ML optimization if CV fails
                op_gp = None

        # If op_gp is not set (CV failed or wasn't used), use current GP or initialize one
        if 'op_gp' not in locals() or op_gp is None:
            if hasattr(self, 'gp'):
                op_gp = self.gp
            else:
                # Create a basic GP with default hyperparameters
                if self.fit_amp:
                    kernel = self.kernel * np.var(_y)
                else:
                    kernel = self.kernel
                op_gp = george.GP(kernel=kernel, fit_mean=self.fit_mean, mean=np.median(_y),
                                white_noise=self.white_noise, fit_white_noise=self.fit_white_noise)
                op_gp.compute(_theta)

        tf = time.time()
        timing = tf - t0
        self.training_results["gp_hyperparam_opt_time"].append(timing)

        return op_gp, timing
        
    
    def eval_gp_at_iteration(self, iter, return_var=False):
        
        # gp_iter = self.gp 
        if (iter == 0) or (len(self.training_results["iteration"]) == 0):
            _theta_cond = self._theta[:self.ninit_train]
            _y_cond = self._y[:self.ninit_train]
            try:
                gp_iter = self.set_hyperparameter_vector(self.gp, self.training_results["gp_hyperparameters"][0])
            except:
                gp_iter = self.set_hyperparameter_vector(self.gp, self.initial_gp_hyperparameters)
        # Handle iter=-1 case to use all data instead of excluding last element
        elif (iter == -1) or (iter == len(self.training_results["iteration"])):
            _theta_cond = self._theta
            _y_cond = self._y
            # Use the latest parameters (last in the list)
            gp_iter = self.set_hyperparameter_vector(self.gp, self.training_results["gp_hyperparameters"][-1])
        elif (iter > 0) and (iter < len(self.training_results["iteration"])):
            _theta_cond = self._theta[:self.ninit_train + iter]
            _y_cond = self._y[:self.ninit_train + iter]
            last_hp = self.training_results["gp_hyperparameters"][iter]
            gp_iter = self.set_hyperparameter_vector(self.gp, last_hp)
        else:
            raise ValueError(f"Iteration {iter} exceeds available training iterations ({self.training_results['iteration'][-1]}).")
            
        gp_iter.compute(_theta_cond)

        def gp_predict(x):
            # Ensure x is properly shaped for george GP
            x = np.atleast_2d(x)
            if x.shape[0] == 1 and x.shape[1] != self.ndim:
                # If we have a single point but wrong shape, transpose
                if x.shape[1] == 1 and x.shape[0] == self.ndim:
                    x = x.T
                elif len(x.flatten()) == self.ndim:
                    x = x.reshape(1, -1)
            return gp_iter.predict(_y_cond, x, return_var=return_var, return_cov=False)
        
        return gp_predict


    def surrogate_log_likelihood(self, theta_xs, iter=-1, return_var=False):
        """
        Evaluate predictive mean of the GP at point(s) ``theta_xs``
        
        This method is vectorized to handle both single parameter vectors and
        arrays of parameter vectors efficiently.

        :param theta_xs: Point(s) to evaluate GP mean at. Can be:
            - 1D array of shape (ndim,) for single point
            - 2D array of shape (npoints, ndim) for multiple points
        :type theta_xs: *array-like*
        :param iter: Iteration number of GP to use. If -1, uses most recent GP.
        :type iter: *int, optional*
        :param return_var: Whether to also return variance predictions.
        :type return_var: *bool, optional*

        :returns: 
            - **ypred** (*array*) -- GP mean(s) evaluated at ``theta_xs``. Shape matches input.
            - **varpred** (*array, optional*) -- GP variance(s) if return_var=True.
        :rtype: *array or tuple of arrays*
        """
        
        # Convert input to numpy array and handle dimensionality
        theta_xs = np.asarray(theta_xs)
        original_shape_1d = False
        
        # Handle 1D input (single point)
        if theta_xs.ndim == 1:
            theta_xs = theta_xs.reshape(1, -1)
            original_shape_1d = True
        elif theta_xs.ndim != 2:
            raise ValueError(f"theta_xs must be 1D or 2D array, got {theta_xs.ndim}D")
        
        # Apply scaling transformation
        _theta_xs = self.theta_scaler.transform(theta_xs)
        
        # Get GP at specified iteration
        if hasattr(self, "training_results") and len(self.training_results["iteration"]) > 0:
            gp_ii = self.eval_gp_at_iteration(iter, return_var=return_var)
        else:
            gp_ii = lambda x: self.gp.predict(self._y, x, return_var=return_var, return_cov=False)

        # Apply the GP and handle return values
        if return_var == False:
            _ypred = gp_ii(_theta_xs)
            ypred = self.y_scaler.inverse_transform(_ypred.reshape(-1, 1)).flatten()
            
            # Return single value if input was 1D, otherwise return array
            if original_shape_1d:
                return ypred[0]
            else:
                return ypred
                
        else:
            _ypred, _varpred = gp_ii(_theta_xs)
            ypred = self.y_scaler.inverse_transform(_ypred.reshape(-1, 1)).flatten()
            varpred = self.y_scaler.inverse_transform(_varpred.reshape(-1, 1)).flatten()

            # Return single values if input was 1D, otherwise return arrays
            if original_shape_1d:
                return ypred[0], varpred[0]
            else:
                return ypred, varpred
    
    
    def surrogate_likelihood(self, theta_xs):
        """
        Evaluate predictive probability (not log-probability) of the GP at point(s) theta_xs
        
        This method is vectorized to handle both single parameter vectors and
        arrays of parameter vectors efficiently.
        
        :param theta_xs: Point(s) to evaluate GP probability at. Can be:
            - 1D array of shape (ndim,) for single point
            - 2D array of shape (npoints, ndim) for multiple points
        :type theta_xs: *array-like*
        
        :returns: GP probability/probabilities evaluated at ``theta_xs``. Shape matches input.
        :rtype: *float or array*
        """
        # Get log-probability from GP (already vectorized)
        log_prob = self.surrogate_log_likelihood(theta_xs)
        
        # Convert to probability (works element-wise for arrays)
        prob = np.exp(log_prob)
        
        return prob


    def create_cached_surrogate_likelihood(self, iter=-1, return_var=False):
        """
        Create a cached surrogate likelihood function that computes the GP once
        and can be evaluated multiple times without recomputing the GP.
        
        This is useful when you need to evaluate the surrogate likelihood at many
        different points with the same GP configuration, as it avoids the expensive
        GP computation (gp.compute()) on each call.
        
        :param iter: Iteration number of GP to use. If -1, uses most recent GP.
        :type iter: *int, optional*
        
        :returns: A cached likelihood function that can be called with theta_xs
        :rtype: *callable*
        """
        
        # Determine training data and hyperparameters for the specified iteration
        if hasattr(self, "training_results") and len(self.training_results["iteration"]) > 0:
            # Handle iter=-1 case to use all data
            if iter == -1:
                _theta_cond = self._theta
                _y_cond = self._y
                hyperparams = self.training_results["gp_hyperparameters"][-1]
            else:
                _theta_cond = self._theta[:self.ninit_train + iter]
                _y_cond = self._y[:self.ninit_train + iter]
                hyperparams = self.training_results["gp_hyperparameters"][-1]
        else:
            _theta_cond = self._theta
            _y_cond = self._y
            hyperparams = self.gp.get_parameter_vector()
        
        # Create a fresh GP instance with the same kernel
        gp_iter = gp_utils.configure_gp(_theta_cond, _y_cond, self.kernel, 
                                        fit_amp=self.fit_amp, 
                                        fit_mean=self.fit_mean,
                                        fit_white_noise=self.fit_white_noise,
                                        white_noise=self.white_noise,
                                        hyperparameters=hyperparams)
        
        # Compute the GP and save so that it doesn't need to be recomputed later
        try:
            gp_iter.compute(_theta_cond)
        except Exception as e:
            print(f"create_cached_surrogate_likelihood: Error computing GP at iteration {iter}: {e}")
            raise
        
        # Return a picklable cached surrogate likelihood object
        return CachedSurrogateLikelihood(gp_iter, _y_cond, self.theta_scaler, 
                                       self.y_scaler, self.ndim, return_var=return_var)


    def find_next_point(self, nopt=3, optimizer_kwargs={}):
        """
        Find next set of ``(theta, y)`` training points by maximizing the
        active learning utility function.

        :param nopt: (*int, optional*) 
            Number of times to restart the objective function optimization. 
            Defaults to 1. Increase to avoid converging to local minima.
        :param optimizer_kwargs: (*dict, optional*)
            Additional keyword arguments passed to scipy optimizer. Default is {}.
        """

        opt_timing_0 = time.time()
        
        predict_gp = lambda _theta_xs: self.gp.predict(self._y, _theta_xs, return_var=True)
        
        # Create objective function with appropriate parameters for different algorithms
        if self.algorithm == "jones":
            # Jones (Expected Improvement) requires y_best parameter
            y_best = np.max(self._y)  # Current best observed value
            obj_fn = partial(self.utility, predict_gp=predict_gp, bounds=self._bounds, y_best=y_best)
        else:
            # Other algorithms (BAPE, AGP, etc.) don't need y_best
            obj_fn = partial(self.utility, predict_gp=predict_gp, bounds=self._bounds)
        
        # Get gradient function if available
        grad_obj_fn = None
        if hasattr(self, 'grad_utility') and self.grad_utility is not None:
            if self.algorithm == "jones":
                grad_obj_fn = partial(self.grad_utility, gp=self.gp, bounds=self._bounds, y_best=y_best)
            else:
                grad_obj_fn = partial(self.grad_utility, gp=self.gp, bounds=self._bounds)

        # Always use serial execution for acquisition function optimization
        _thetaN, _ = ut.minimize_objective(obj_fn, 
                                        bounds=self._bounds,
                                        nopt=nopt,
                                        ps=self._prior_sampler,
                                        method=self.obj_opt_method,
                                        options=optimizer_kwargs,
                                        grad_obj_fn=grad_obj_fn,
                                        pool=None)

        opt_timing = time.time() - opt_timing_0
        # self.training_results["acquisition_optimizer_niter"].append(opt_result.nit)
        
        # Validate optimization result
        if not np.all(np.isfinite(_thetaN)):
            print(f"Warning: Acquisition function optimization failed. Falling back to random sampling.")
            # Fall back to random sampling from prior
            _thetaN = self._prior_sampler(nsample=1).flatten()
        
        thetaN = self.theta_scaler.inverse_transform(_thetaN.reshape(1, -1))
        yN = self.true_log_likelihood(thetaN.flatten())

        # Validate new training point
        if not np.any(np.isfinite(thetaN.flatten())):
            print(f"New theta contains NaN or Inf: {thetaN}")
            return None, None, opt_timing
        if not np.isfinite(float(yN)):
            print(f"New y value is NaN or Inf: {yN}. Check your likelihood function at theta={thetaN}")
            return None, None, opt_timing
        
        # add theta and y to training samples
        theta_prop = np.append(self.theta(), thetaN, axis=0)
        y_prop = np.append(self.y(), yN)
        
        # refit scaling functions including new point
        _theta_prop, _y_prop = self.refit_scalers(theta_prop, y_prop)
                
        if _theta_prop.shape[0] != _y_prop.shape[0]:
            return None, None, opt_timing
        
        if not np.all(np.isfinite(_theta_prop)):
            print("_theta_prop contains NaN or Inf after scaling. Check training data and scalers.")
            return None, None, opt_timing
        if not np.all(np.isfinite(_y_prop)):
            print("_y_prop contains NaN or Inf after scaling. Check training data and scalers.")
            return None, None, opt_timing

        return _theta_prop, _y_prop, opt_timing


    def active_train(self, niter=100, algorithm="bape", gp_opt_freq=20, save_progress=False,
                     obj_opt_method="l-bfgs-b", nopt=5, optimizer_kwargs={}, use_grad_opt=True,
                     show_progress=True, allow_opt_multiproc=True, max_attempts=10): 
        """
        Perform active learning to iteratively improve the surrogate model.
        
        Uses acquisition functions to intelligently select new training points that
        will most improve the Gaussian Process model. Different algorithms balance
        exploration (uncertainty reduction) vs exploitation (finding optima).
        
        :param niter: (*int, optional, default=100*)
            Number of active learning iterations. Each iteration adds one new training point.
            
        :param algorithm: (*str, optional, default="bape"*)
            Active learning algorithm. Options:
            - "bape": Bayesian Active Parameter Estimation (exploration-focused)
            - "jones": Jones algorithm (exploitation-focused, good for optimization)
            - "agp": Augmented Gaussian Process (balanced)
            - "alternate": Alternates between exploration and exploitation
            
        :param gp_opt_freq: (*int, optional, default=20*)
            Frequency of GP hyperparameter re-optimization. GP hyperparameters are
            re-optimized every gp_opt_freq iterations. Lower values = more optimization.
            
        :param save_progress: (*bool, optional, default=False*)
            Whether to save training progress data for later analysis.
            
        :param obj_opt_method: (*str, optional, default="nelder-mead"*)
            Optimization method for acquisition function. Options:
            - "l-bfgs-b": L-BFGS-B (good with gradients)
            - "nelder-mead": Nelder-Mead simplex (gradient-free)
            
        :param nopt: (*int, optional, default=1*)
            Number of optimization restarts for acquisition function. Higher values
            help avoid local minima but increase computation time.
            
        :param use_grad_opt: (*bool, optional, default=True*)
            Whether to use gradient information if available. Set False for
            gradient-free optimization.
            
        :param optimizer_kwargs: (*dict, optional, default={}*)
            Additional keyword arguments passed to the optimizer.
            
        :param show_progress: (*bool, optional, default=True*)
            Whether to display progress bar during training.
            
        .. note::
        
            Active learning algorithms have different purposes:
            
            - **BAPE**: Best for uncertainty quantification and space-filling
            - **Jones**: Best for finding likelihood maxima/minima (optimization)  
            - **Alternate**: Good balance for both exploration and exploitation
            - **AGP**: Another balanced approach
            
            The method automatically handles GP re-training and hyperparameter optimization
            based on the specified frequency. Training data is accumulated in _theta and _y
            attributes.
        
        .. code-block:: python

        Basic active learning with BAPE:
        
        >>> sm.active_train(niter=50, algorithm="bape")
        
        Optimization-focused active learning:
        
        >>> sm.active_train(niter=30, algorithm="jones", gp_opt_freq=10)
        
        Balanced approach with frequent GP optimization:
        
        >>> sm.active_train(niter=40, algorithm="alternate", gp_opt_freq=5)
        """

        # Set algorithm
        self.algorithm = str(algorithm).lower()
        self.utility, self.grad_utility = ut.assign_utility(self.algorithm)
        if use_grad_opt == False:
            self.grad_utility = None

        # GP hyperparameter optimization frequency
        self.gp_opt_freq = gp_opt_freq

        # Objective function optimization method
        self.obj_opt_method = obj_opt_method

        if len(self.training_results["iteration"]) == 0:
            first_iter = 0
        else:
            first_iter = self.training_results["iteration"][-1]

        if self.verbose:
            print(f"Running {niter} active learning iterations using {self.algorithm}...")

        # Create iterator with or without progress bar based on show_progress parameter
        iterator = tqdm.tqdm(range(1, niter+1)) if show_progress else range(1, niter+1)
        
        for ii in iterator:

            attempts = 0
            success = False
            while not success and attempts < max_attempts:

                # Find next training point! (always single-threaded)
                _theta_prop, _y_prop, opt_timing = self.find_next_point(nopt=nopt, optimizer_kwargs=optimizer_kwargs)
                
                if _theta_prop is None or _y_prop is None:
                    attempts += 1
                else:
                    # Fit GP with new training point
                    self.gp, fit_gp_timing = self._fit_gp(_theta=_theta_prop, _y=_y_prop, hyperparameters=self.gp.get_parameter_vector())
                    success = True

                if attempts >= max_attempts:
                    raise RuntimeError(f"Failed to find a valid training point after {max_attempts} attempts. \
                                        Check your likelihood function and training data for issues or increase max_attempts.")
                    
            # If proposed (theta, y) did not cause fitting issues, save to surrogate model obj
            self._theta = _theta_prop
            self._y = _y_prop
            
            # Optimize GP?
            if (ii + first_iter) % self.gp_opt_freq == 0:

                reopt_kwargs = self.opt_gp_kwargs.copy()
                reopt_kwargs["multi_proc"] = allow_opt_multiproc
                # re-optimize hyperparamters
                self.gp, _ = self._opt_gp(**reopt_kwargs)
                
                # record which iteration hyperparameters were optimized
                self.training_results["gp_hyperparameter_opt_iteration"].append(ii + first_iter)
                
                if (save_progress == True) and (ii != 0):
                    self.save()
                    self.plot(plots=["gp_error", "gp_hyperparam"])
                    if self.ndim == 2:
                        self.plot(plots=["gp_fit_2D"])
                    else:
                        self.plot(plots=["gp_train_scatter"])
            
            # evaluate gp training error (scaled)
            try:
                _ypred = self.gp.predict(_y_prop, _theta_prop, return_cov=False, return_var=False)
                ypred = self.y_scaler.inverse_transform(_ypred.reshape(-1, 1)).flatten()
                training_mse = np.mean((self.y() - ypred)**2)
                training_scaled_mse = training_mse / np.var(self.y())
                
                # if hyperparameters were reoptimized, report train error
                if ((ii + first_iter) % self.gp_opt_freq == 0) & self.verbose:
                    print("Train MSE:", training_mse)
            except Exception as e:
                print(f"Warning: Error evaluating GP training error at iteration {ii + first_iter}: {e}")
                training_mse = np.nan
                training_scaled_mse = np.nan

            # evaluate gp test error (scaled)
            if hasattr(self, '_theta_test') and hasattr(self, '_y_test') and len(self._y_test) > 0:
                try:
                    _ytest = self.gp.predict(self._y, self._theta_test, return_cov=False, return_var=False)
                    ytest = self.y_scaler.inverse_transform(_ytest.reshape(-1, 1)).flatten()
                    ytest_true = self.y_scaler.inverse_transform(self._y_test.reshape(-1, 1)).flatten()
                    test_mse = np.mean((ytest_true - ytest)**2)
                    test_scaled_mse = test_mse / np.var(self.y())
                    
                    # if hyperparameters were reoptimized, report test error
                    if ((ii + first_iter) % self.gp_opt_freq == 0) & self.verbose:
                        print("Test MSE:", test_mse)
                except Exception as e:
                    print(f"Warning: Error evaluating GP test error at iteration {ii + first_iter}: {e}")
                    test_mse = np.nan
                    test_scaled_mse = np.nan
            else:
                test_mse = np.nan
                test_scaled_mse = np.nan

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

        if self.cache:
            self.save()
            
            
    def active_train_parallel(self, niter=100, nchains=4, algorithm="bape", gp_opt_freq=20, 
                                   obj_opt_method="nelder-mead", nopt=1, 
                                   use_grad_opt=True, optimizer_kwargs={}, 
                                   show_progress=True):
        """
        Run multiple active learning chains in parallel.
        
        :param niter: (*int, optional*) 
            Number of iterations per chain. Default 100.
            
        :param nchains: (*int, optional*) 
            Number of parallel chains to run. Default 4.
            
        :param algorithm: (*str, optional*) 
            Active learning algorithm. Default "bape".
            
        :param gp_opt_freq: (*int, optional*)
            Frequency of GP hyperparameter optimization. Default 20.
            
        :param obj_opt_method: (*str, optional*)
            Optimization method for acquisition function. Default "nelder-mead".
            
        :param nopt: (*int, optional*)
            Number of restarts for acquisition optimization. Default 1.
            
        :param use_grad_opt: (*bool, optional*)
            Whether to use gradient-based optimization. Default True.
            
        :param optimizer_kwargs: (*dict, optional*)
            Additional optimizer kwargs. Default {}.
            
        :param show_progress: (*bool, optional*) 
            Whether to display progress bar during parallel chain execution. Default is True.
            
        Notes
        -----
        This function uses multiprocessing.Pool instead of threading, which can provide
        better performance for CPU-intensive tasks and avoids GIL limitations. However:
        
        - All model data must be pickleable (which it should be for SurrogateModel)
        - Each process runs in separate memory space (higher memory usage)
        - Process startup overhead is higher than threading
        - Better isolation between chains (one chain failure won't affect others)
        - Can achieve true parallelism on multi-core systems
        
        The function automatically respects the ncore limit and won't create more processes
        than specified in self.ncore.
        """
        
        
        # Set algorithm attribute to avoid save issues
        self.algorithm = str(algorithm).lower()
        
        # Initialize training results if not present
        if not hasattr(self, 'training_results') or not self.training_results:
            self.training_results = {"iteration" : [], 
                                   "gp_hyperparameters" : [],  
                                   "training_mse" : [],
                                   "test_mse" : [],
                                   "training_scaled_mse" : [],
                                   "test_scaled_mse" : [],
                                   "gp_kl_divergence" : [],
                                   "gp_train_time" : [],
                                   "obj_fn_opt_time" : [],
                                   "gp_hyperparameter_opt_iteration" : [],
                                   "gp_hyperparam_opt_time" : [],
                                   "acquisition_optimizer_niter": []}
        
        if self.verbose:
            print(f"\nRunning {nchains} parallel active learning chains for {niter} iterations each...")
            print(f"Algorithm: {algorithm}, Method: {obj_opt_method}")
            print(f"Using multiprocessing with max {min(nchains, self.ncore)} processes")
        
        # Store original training data
        original_ntrain = self.ntrain
        
        # Track results from all chains
        all_new_theta = []
        all_new_y = []
        chain_results = []
        
        # Determine number of processes to use (respect ncore limit)
        max_processes = min(nchains, self.ncore)
        use_multiprocessing = (nchains > 1) and (self.ncore > 1)
        
        if use_multiprocessing:
            if self.verbose:
                print(f"Running with {max_processes} processes (limited by ncore={self.ncore})")
        
        try:
            if use_multiprocessing:
                # Prepare arguments for each chain
                chain_args = []
                for i in range(nchains):
                    # Create a pickleable state dictionary for each chain
                    chain_state = self._get_pickleable_state()
                    chain_state['chain_id'] = i
                    chain_state['savedir'] = f"{self.savedir}/chain_{i}"
                    
                    args = (
                        chain_state,
                        niter,
                        algorithm,
                        gp_opt_freq,
                        obj_opt_method,
                        nopt,
                        use_grad_opt,
                        optimizer_kwargs,
                    )
                    chain_args.append(args)
                
                pool = self._get_pool(ncore=max_processes)
                
                if show_progress:
                    results = []
                    with tqdm.tqdm(total=nchains, desc="Running parallel chains (MP)") as pbar:
                        for result in pool.imap(_run_chain_worker_mp, chain_args):
                            results.append(result)
                            pbar.update(1)
                else:
                    # Use map for simpler execution
                    results = pool.map(_run_chain_worker_mp, chain_args)
                    
                self._close_pool(pool)
                
                # Process results
                for i, result in enumerate(results):
                    if result is not None and len(result) == 3:
                        new_theta, new_y, training_results = result
                        all_new_theta.append(new_theta)
                        all_new_y.append(new_y)
                        chain_results.append(training_results)
                    else:
                        if self.verbose:
                            print(f"Chain {i} failed or returned invalid result")
                        all_new_theta.append(np.array([]).reshape(0, self.ndim))
                        all_new_y.append(np.array([]))
                        chain_results.append({})
            
            else:
                # Fallback to sequential execution
                if self.verbose:
                    print("Running chains sequentially...")
                
                if show_progress:
                    sequential_progress = tqdm.tqdm(total=nchains, desc="Running chains sequentially")
                
                for i in range(nchains):
                    if self.verbose:
                        print(f"Running chain {i+1}/{nchains}...")
                    
                    result = self._run_chain_worker(i, niter=niter, algorithm=algorithm, 
                                                  gp_opt_freq=gp_opt_freq, obj_opt_method=obj_opt_method,
                                                  nopt=nopt, use_grad_opt=use_grad_opt, 
                                                  optimizer_kwargs=optimizer_kwargs, show_progress=False,
                                                  allow_opt_multiproc=False)
                    
                    if result is not None:
                        new_theta, new_y, training_results = result
                        all_new_theta.append(new_theta)
                        all_new_y.append(new_y)
                        chain_results.append(training_results)
                    else:
                        if self.verbose:
                            print(f"Chain {i} failed")
                        all_new_theta.append(np.array([]).reshape(0, self.ndim))
                        all_new_y.append(np.array([]))
                        chain_results.append({})
                    
                    # Update sequential progress bar
                    if show_progress:
                        sequential_progress.update(1)
                
                # Close sequential progress bar
                if show_progress:
                    sequential_progress.close()
                    
        except Exception as e:
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_line = traceback.extract_tb(exc_traceback)[-1]
            print(f"Multiprocessing execution failed ({e}) line {tb_line}.")
            raise e
        
        # Combine all new training samples
        if self.verbose:
            print("\nCombining training samples from all chains...")
        
        self._combine_chain_results(all_new_theta, all_new_y, chain_results)
        
        if self.verbose:
            total_new_samples = sum(len(theta) for theta in all_new_theta)
            print(f"Successfully combined {total_new_samples} new training samples from {nchains} chains")
            print(f"Total training samples: {self.ntrain} (was {original_ntrain})")
        
        # Final GP optimization with all combined data
        if self.verbose:
            print("\nPerforming final GP optimization with combined dataset...")
        self.gp, _ = self._opt_gp(**self.opt_gp_kwargs)
        
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

        if (self.like_fn_name == "surrogate") and (not hasattr(self, 'gp')):
            raise NameError("GP has not been trained")

        if not hasattr(self, 'prior_fn'):
            raise NameError("prior_fn has not been specified")
        
        if not hasattr(self, 'like_fn'):
            self.like_fn = self.surrogate_log_likelihood

        theta = np.asarray(theta).reshape(1,-1)

        lnp = self.like_fn(theta) + self.prior_fn(theta)

        return lnp


    def find_map(self, theta0=None, prior_fn=None, method="nelder-mead", nRestarts=15, options=None):

        raise NotImplementedError("Not implemented.")


    def run_emcee(self, like_fn=None, prior_fn=None, p0=None, nwalkers=None, nsteps=int(5e4), sampler_kwargs={}, run_kwargs={},
                  opt_init=False, multi_proc=True, prior_fn_comment=None, burn=None, thin=None, samples_file=None, min_ess=int(1e4)):
        """
        Sample the posterior using the emcee affine-invariant MCMC algorithm.
        
        This method uses the emcee package to perform Markov Chain Monte Carlo (MCMC) 
        sampling on either the trained GP surrogate model or the true likelihood function.
        The affine-invariant ensemble sampler is robust and works well for a wide variety
        of posterior shapes without requiring manual tuning of step sizes.
        
        :param like_fn: (*callable, str, or None, optional*)
            Likelihood function to sample. Options:
            - None (default): Uses the trained GP surrogate model (self.surrogate_log_likelihood)
            - "surrogate", "gp": Uses the GP surrogate model explicitly
            - "true": Uses the true likelihood function (self.true_log_likelihood)
            - callable: Custom likelihood function with signature like_fn(theta)
            Default is None.
            
        :param prior_fn: (*callable or None, optional*)
            Log-prior function with signature prior_fn(theta). Should return log-probability
            density. If None, uses uniform prior with bounds from self.bounds.
            Default is None.
            
        :param nwalkers: (*int or None, optional*)
            Number of MCMC walkers in the ensemble. Should be at least 2*ndim.
            If None, defaults to 10*ndim. More walkers improve convergence but increase
            computational cost. Default is None.
            
        :param nsteps: (*int, optional*)
            Number of MCMC steps per walker. Total number of likelihood evaluations
            will be nwalkers * nsteps. Default is 50000.
            
        :param sampler_kwargs: (*dict, optional*)
            Additional keyword arguments passed to emcee.EnsembleSampler constructor.
            Common options include:
            - 'a': Stretch move scale parameter (default: 2.0)
            - 'moves': Custom proposal moves
            Default is {}.
            
        :param run_kwargs: (*dict, optional*)
            Additional keyword arguments passed to the run_mcmc() method.
            Common options include:
            - 'progress': Show progress bar (default: True)
            - 'store': Store chain in memory (default: True)
            Default is {}.
            
        :param opt_init: (*bool, optional*)
            Whether to initialize walkers near the maximum a posteriori (MAP) estimate.
            If True, uses find_map() to locate starting point. If False, initializes
            walkers randomly from the prior. Default is False.
            
        :param multi_proc: (*bool, optional*)
            Whether to use multiprocessing with self.ncore processes. Generally
            recommended for expensive likelihood evaluations. Default is True.
            
        :param prior_fn_comment: (*str or None, optional*)
            Comment describing the prior function for logging purposes. If None
            and prior_fn is provided, attempts to extract function name.
            Default is None.
            
        :param burn: (*int or None, optional*)
            Number of burn-in samples to discard from each walker. If None, 
            automatically estimates burn-in using autocorrelation analysis.
            Default is None.
            
        :param thin: (*int or None, optional*)
            Thinning factor - keep every thin-th sample to reduce autocorrelation.
            If None, automatically estimates based on autocorrelation time.
            Default is None.
            
        :param samples_file: (*str or None, optional*)
            If provided, saves the final samples to this file in NumPy .npz format.
            Default is None.
            
        :param min_ess: (*int, optional*)
            Minimum effective sample size. If the number of final samples is less than 
            min_ess, will run additional sampling rounds and combine samples until the 
            total number of samples exceeds min_ess. Default is 0 (no minimum required).
            
        Attributes Set
        --------------
        sampler : emcee.EnsembleSampler
            The emcee sampler object containing full chain and metadata
        emcee_samples : ndarray of shape (nsamples_final, ndim)
            Final MCMC samples after burn-in and thinning
        emcee_samples_full : ndarray of shape (nsteps, nwalkers, ndim)
            Full MCMC chain before processing
        emcee_samples_true : ndarray of shape (nsamples_final, ndim)
            Final samples when using true likelihood (like_fn="true")
        emcee_samples_gp : ndarray of shape (nsamples_final, ndim)
            Final samples when using surrogate likelihood
        emcee_run : bool
            Flag indicating emcee has been successfully run
        emcee_runtime : float
            Wall-clock time taken for emcee sampling in seconds
        nwalkers : int
            Number of walkers used
        nsteps : int
            Number of steps per walker
        burn : int
            Burn-in length used for final samples
        thin : int
            Thinning factor used for final samples
        iburn : int
            Automatically estimated burn-in length
        ithin : int
            Automatically estimated thinning factor
        acc_frac : float
            Mean acceptance fraction across all walkers
        autcorr_time : float
            Mean autocorrelation time in steps
        like_fn_name : str
            Name of likelihood function used ("true", "surrogate", or "likelihood")
        prior_fn_comment : str
            Description of prior function used
            
        
        .. code-block:: python
        
        Sample surrogate model with default settings:
        
        >>> sm.run_emcee()
        
        Sample true likelihood with specific settings:
        
        >>> sm.run_emcee(like_fn="true", nwalkers=100, nsteps=10000)
        
        Use custom prior and optimize initialization:
        
        >>> def log_prior(theta):
        ...     # Custom Gaussian prior
        ...     return -0.5 * np.sum((theta/2)**2)
        >>> sm.run_emcee(prior_fn=log_prior, opt_init=True)
        
        Run with manual burn-in and thinning:
        
        >>> sm.run_emcee(nsteps=100000, burn=10000, thin=10)
        
        References
        ----------
        emcee documentation: https://emcee.readthedocs.io/
        Foreman-Mackey et al. (2013): "emcee: The MCMC Hammer", PASP, 125, 306-312
        """

        import emcee

        if like_fn is None:
            print("Initializing emcee with self.surrogate_log_likelihood surrogate model as likelihood.")
            self.like_fn_name = "surrogate"
            self.like_fn = self.surrogate_log_likelihood
        else:
            self.like_fn = like_fn
            self.like_fn_name = "likelihood"

        if prior_fn is None:
            print(f"No prior_fn specified. Defaulting to uniform prior with bounds {self.bounds}")
            self.prior_fn = partial(ut.lnprior_uniform, bounds=self.bounds)

            # Comment for output log file
            self.prior_fn_comment =  f"Default uniform prior. \n" 
            self.prior_fn_comment += f"Prior function: ut.prior_fn_uniform\n"
            self.prior_fn_comment += f"\twith bounds {self.bounds}"

        else:
            self.prior_fn = prior_fn

            # Comment for output log file
            if prior_fn_comment is None:
                self.prior_fn_comment = f"User defined prior."
                try:
                    self.prior_fn_comment += f"Prior function: {self.prior_fn.__name__}"
                except:
                    self.prior_fn_comment += "Prior function: unrecorded"
            else:
                self.prior_fn_comment = prior_fn_comment

        # number of walkers, and number of steps per walker
        if nwalkers is None:
            self.nwalkers = int(10 * self.ndim)
        else:
            self.nwalkers = int(nwalkers)
        self.nsteps = int(nsteps)

        # Initialize walker positions
        if p0 is not None:
            pass  # use caller-provided p0
        elif opt_init:
            p0 = self.find_map(prior_fn=self.prior_fn)
        else:
            p0 = ut.prior_sampler(nsample=self.nwalkers, bounds=self.bounds, sampler="uniform", random_state=None)

        # set up multiprocessing pool with MPI safety
        emcee_pool = self._get_pool(ncore=self.ncore) if multi_proc and self.ncore > 1 else None
        emcee_ncore = self.ncore if (multi_proc and self.ncore > 1) else 1
            
        if self.verbose:
            print(f"Running emcee with {self.nwalkers} walkers for {self.nsteps} steps on {emcee_ncore} cores...")

        # Multi-run setup for minimum effective sample size
        all_chains = []
        all_run_times = []
        accumulated_samples = 0
        run_number = 1
        iMinSamples = max(min_ess, 1)

        while accumulated_samples < iMinSamples:
            if min_ess > 0:
                print(f"\nRun {run_number}: Need {max(0, iMinSamples - accumulated_samples)} more samples...")
                print("="*50)

            # Run the sampler!
            emcee_t0 = time.time()
            self.emcee_sampler = emcee.EnsembleSampler(self.nwalkers, 
                                                 self.ndim, 
                                                 self.lnprob, 
                                                 pool=emcee_pool,
                                                 **sampler_kwargs)

            self.emcee_sampler.run_mcmc(p0, self.nsteps, progress=True, **run_kwargs)
            
            # close pool 
            self._close_pool(emcee_pool)

            # record emcee runtime
            current_runtime = time.time() - emcee_t0
            all_run_times.append(current_runtime)

            # burn, thin, and flatten samples for this run
            current_iburn, current_ithin = mcmc_utils.estimate_burnin(self.emcee_sampler, verbose=self.verbose)
            current_samples_full = self.emcee_sampler.get_chain()

            current_burn = burn if burn is not None else current_iburn
            current_thin = thin if thin is not None else current_ithin

            current_samples = self.emcee_sampler.get_chain(discard=current_burn, thin=current_thin, flat=True)
            all_chains.append(current_samples)
            
            current_nsamples = current_samples.shape[0]
            accumulated_samples += current_nsamples

            if min_ess > 0:
                print(f"Run {run_number} complete: {current_nsamples} samples")
                print(f"Total accumulated samples: {accumulated_samples}")

            # If min_ess requirement met, break
            if accumulated_samples >= iMinSamples:
                break

            run_number += 1
            
            # Reset initial positions for next run (start from end of previous run)
            if run_number <= 10:  # Limit to prevent infinite loops
                # Use final positions from this run as starting positions for next run
                p0 = self.emcee_sampler.get_last_sample().coords
            else:
                print(f"WARNING: Reached maximum of 10 runs, stopping with {accumulated_samples} samples")
                break
        
        # Combine all chains and compute overall statistics
        if len(all_chains) > 1:
            self.emcee_samples = np.vstack(all_chains)
            print(f"\nCombined {len(all_chains)} runs into {self.emcee_samples.shape[0]} total samples")
        else:
            self.emcee_samples = all_chains[0]
        
        # Use final run for chain statistics and samples_full
        self.emcee_samples_full = current_samples_full
        self.iburn = current_iburn
        self.ithin = current_ithin 
        self.burn = current_burn
        self.thin = current_thin
        self.emcee_runtime = sum(all_run_times) 
        
        if self.like_fn_name == "true":
            self.emcee_samples_true = self.emcee_samples
        elif self.like_fn_name == "surrogate":
            self.emcee_samples_gp = self.emcee_samples

        # get acceptance fraction and autocorrelation time
        self.acc_frac = np.mean(self.emcee_sampler.acceptance_fraction)
        try:
            self.autcorr_time = np.mean(self.emcee_sampler.get_autocorr_time())
        except Exception:
            self.autcorr_time = np.mean(self.emcee_sampler.get_autocorr_time(quiet=True))
        if self.verbose:
            print(f"Total samples: {self.emcee_samples.shape[0]}")
            print("Mean acceptance fraction: {0:.3f}".format(self.acc_frac))
            print("Mean autocorrelation time: {0:.3f} steps".format(self.autcorr_time))

        # record that emcee has been run
        self.emcee_run = True

        if self.cache:
            try:
                self.save()
            except:
                pass

        if samples_file is not None:
            fname = f"{self.savedir}/{samples_file}"
        else:
            if self.like_fn_name == "true":
                fname = f"{self.savedir}/emcee_samples_final_{self.like_fn_name}.npz"
            else:
                if len(self.training_results["iteration"]) == 0:
                    current_iter = 0
                else:   
                    current_iter = self.training_results["iteration"][-1]
                fname = f"{self.savedir}/emcee_samples_final_{self.like_fn_name}_iter_{current_iter}.npz"
        print(f"Saving final emcee samples to {fname} ...")
        np.savez(fname, samples=self.emcee_samples)
            

    def run_dynesty(self, like_fn=None, prior_transform=None, mode="dynamic", sampler_kwargs={}, run_kwargs={},
                    multi_proc=False, save_iter=None, prior_transform_comment=None, samples_file=None, min_ess=int(1e4)):
        """
        Sample the posterior using the dynesty nested sampling algorithm.
        
        This method uses the dynesty package to perform nested sampling on either the 
        trained GP surrogate model or the true likelihood function. Dynesty is particularly 
        effective for estimating the Bayesian evidence and exploring multi-modal posteriors.
        
        :param like_fn: (*callable, str, or None, optional*)
            Likelihood function to sample. Options:
            - None (default): Uses the trained GP surrogate model (self.surrogate_log_likelihood)
            - "surrogate", "gp": Uses the GP surrogate model explicitly
            - "true": Uses the true likelihood function (self.true_log_likelihood)  
            - callable: Custom likelihood function with signature like_fn(theta)
            Default is None.
            
        :param prior_transform: (*callable or None, optional*)
            Prior transformation function that maps from unit hypercube [0,1]^ndim
            to the parameter space. Should have signature prior_transform(u) where
            u is array of shape (ndim,) with values in [0,1]. If None, uses uniform
            prior with bounds from self.bounds. Default is None.
            
        :param mode: (*{"dynamic", "static"}, optional*)
            Dynesty sampling mode. "dynamic" uses DynamicNestedSampler which adaptively
            allocates live points, while "static" uses fixed number of live points.
            Dynamic mode is generally more efficient. Default is "dynamic".
            
        :param sampler_kwargs: (*dict, optional*)
            Additional keyword arguments passed to the dynesty sampler constructor.
            Common options include:
            - 'nlive': Number of live points (default: 50*ndim)
            - 'bound': Bounding method ('multi', 'single', 'none')
            - 'sample': Sampling method ('auto', 'unif', 'rwalk', 'slice', 'rslice', 'hslice')
            Default is {}.
            
        :param run_kwargs: (*dict, optional*)
            Additional keyword arguments passed to the run_nested() method.
            Common options include:
            - 'dlogz': Target evidence uncertainty (default: 0.5)
            - 'maxiter': Maximum number of iterations (default: 50000)
            - 'wt_kwargs': Weight function arguments (default: {'pfrac': 1.0})
            - 'stop_kwargs': Stopping criterion arguments (default: {'pfrac': 1.0})
            Default is {}.
            
        :param multi_proc: (*bool, optional*)
            Whether to use multiprocessing. If True, uses self.ncore processes.
            Note that multiprocessing can sometimes be slower due to overhead.
            Default is False.
            
        :param save_iter: (*int or None, optional*)
            If provided, saves the sampler state every save_iter iterations to allow
            for checkpointing and resuming long runs. Saves to 
            '{savedir}/dynesty_sampler_{like_fn_name}.pkl'. Default is None.
            
        :param prior_transform_comment: (*str or None, optional*)
            Comment describing the prior transform for logging purposes. If None
            and prior_transform is provided, attempts to extract function name.
            Default is None.
            
        :param samples_file: (*str or None, optional*)
            If provided, saves the final samples to this file in NumPy .npz format.
            Default is None.
            
        :param min_ess: (*int, optional*)
            Minimum effective sample size. If the number of final samples is less than 
            min_ess, will run additional sampling rounds and combine samples until the 
            total number of samples exceeds min_ess. Default is 0 (no minimum required).
            
        Attributes Set
        --------------
        res : dynesty.results.Results
            Complete dynesty results object containing samples, weights, evidence, etc.
        dynesty_samples : ndarray of shape (nsamples, ndim)
            Resampled posterior samples with equal weights
        dynesty_samples_true : ndarray of shape (nsamples, ndim)
            Posterior samples when using true likelihood (like_fn="true")
        dynesty_samples_surrogate : ndarray of shape (nsamples, ndim)  
            Posterior samples when using surrogate likelihood
        dynesty_run : bool
            Flag indicating dynesty has been successfully run
        dynesty_runtime : float
            Wall-clock time taken for dynesty sampling in seconds
        like_fn_name : str
            Name of likelihood function used ("true", "surrogate", or "custom")
        prior_transform_comment : str
            Description of prior transform used
            
        Notes
        -----
        Dynesty is particularly well-suited for:
        - Computing Bayesian evidence for model comparison
        - Exploring multi-modal posteriors
        - Providing robust posterior sampling without tuning
        
        The default settings prioritize posterior sampling over evidence estimation
        by setting pfrac=1.0, which focuses computational effort on high-likelihood
        regions rather than exploring the full prior volume.
        
        Examples
        --------
        Sample surrogate model with default settings:
        
        >>> sm.run_dynesty()
        
        Sample true likelihood with more live points:
        
        >>> sm.run_dynesty(like_fn="true", sampler_kwargs={'nlive': 1000})
        
        Use custom prior with bounds [-5, 5] for each parameter:
        
        >>> def my_prior(u):
        ...     return 10*u - 5  # maps [0,1] to [-5,5]
        >>> sm.run_dynesty(prior_transform=my_prior)
        
        Run with checkpointing every 1000 iterations:
        
        >>> sm.run_dynesty(save_iter=1000, run_kwargs={'maxiter': 50000})
        
        References
        ----------
        Dynesty documentation: https://dynesty.readthedocs.io/
        Speagle (2020): "dynesty: a dynamic nested sampling package for estimating
        Bayesian posteriors and evidences", MNRAS, 493, 3132-3158
        """

        import dynesty
        from dynesty import NestedSampler
        from dynesty import DynamicNestedSampler
        from dynesty import utils as dyfunc
        
        # Determine likelihood function and name
        if like_fn is None:
            print("Initializing dynesty with self.surrogate_log_likelihood surrogate model as likelihood.")
            self.like_fn_name = "surrogate"
            self.like_fn = self.surrogate_log_likelihood
        elif callable(like_fn):
            # like_fn is a function/callable
            if like_fn == self.surrogate_log_likelihood:
                print("Initializing dynesty with self.surrogate_log_likelihood surrogate model as likelihood.")
                self.like_fn_name = "surrogate"
                self.like_fn = self.surrogate_log_likelihood
            elif like_fn == self.true_log_likelihood:
                print("Initializing dynesty with self.true_log_likelihood as likelihood.")
                self.like_fn_name = "true"
                self.like_fn = like_fn
            else:
                # Custom function provided
                print("Initializing dynesty with user-provided likelihood function.")
                self.like_fn_name = "custom"
                self.like_fn = like_fn
        elif isinstance(like_fn, str):
            # like_fn is a string
            like_fn_lower = like_fn.lower()
            if like_fn_lower in ["surrogate", "gp", "surrogate_log_likelihood"]:
                print("Initializing dynesty with self.surrogate_log_likelihood surrogate model as likelihood.")
                self.like_fn_name = "surrogate"
                self.like_fn = self.surrogate_log_likelihood
            elif like_fn_lower in ["true", "true_log_likelihood"]:
                print("Initializing dynesty with self.true_log_likelihood as likelihood.")
                self.like_fn_name = "true"
                self.like_fn = self.true_log_likelihood
            else:
                raise ValueError(f"Unknown string identifier for like_fn: '{like_fn}'. "
                                f"Valid options: 'surrogate', 'true', 'gp', 'surrogate_log_likelihood', 'true_log_likelihood'")
        else:
            raise TypeError(f"like_fn must be None, a string, or a callable function. "
                        f"Received type: {type(like_fn)}")
            
        # set up prior transform
        if prior_transform is None:
            self.prior_transform = partial(ut.prior_transform_uniform, bounds=self.bounds)

            # Comment for output log file
            self.prior_transform_comment =  f"Default uniform prior transform. \n" 
            self.prior_transform_comment += f"Prior function: ut.prior_transform_uniform\n"
            self.prior_transform_comment += f"\twith bounds {self.bounds}"
        
        else:
            self.prior_transform = prior_transform

            # Comment for output log file
            if prior_transform_comment is None:
                self.prior_transform_comment = f"User defined prior transform."
                try:
                    self.prior_transform_comment += f"Prior function: {self.prior_transform.__name__}"
                except:
                    self.prior_transform_comment += "Prior function: unrecorded"
            else:
                self.prior_transform_comment = prior_transform_comment

        # start timing dynesty
        dynesty_t0 = time.time()
        
        # set up sampler kwargs
        default_sampler_kwargs = {"bound": "multi",
                                  "nlive": 50*self.ndim,
                                  "sample": "auto"}
        
        for key in default_sampler_kwargs:
            if key not in sampler_kwargs:
                sampler_kwargs[key] = default_sampler_kwargs[key]
        
        # set up multiprocessing pool 
        # default to false. multiprocessing usually slower for some reason
        dynesty_pool = self._get_pool(ncore=self.ncore) if multi_proc and self.ncore > 1 else None
        sampler_kwargs["pool"] = dynesty_pool
        sampler_kwargs["queue_size"] = self.ncore if (multi_proc and self.ncore > 1) else 1
        dynesty_ncore = sampler_kwargs["queue_size"]

        # initialize our nested sampler
        if mode == "dynamic":
            dsampler = DynamicNestedSampler(self.like_fn, 
                                            self.prior_transform, 
                                            self.ndim,
                                            **sampler_kwargs)
            print("Initialized dynesty DynamicNestedSampler.")
        elif mode == "static":
            dsampler = NestedSampler(self.like_fn, 
                                     self.prior_transform, 
                                     self.ndim,
                                     **sampler_kwargs)
            print("Initialized dynesty NestedSampler.")
        else:
            raise ValueError(f"mode {mode} is not a valid option. Choose 'dynamic' or 'static'.")
        
        # set up run kwargs with mode-appropriate defaults
        if mode == "dynamic":
            default_run_kwargs = {"wt_kwargs": {'pfrac': 1.0},
                                  "stop_kwargs": {'pfrac': 1.0},
                                  "maxiter": int(5e4),
                                  "dlogz_init": 0.5}
        else:
            default_run_kwargs = {"maxiter": int(5e4),
                                  "dlogz": 0.5}
        for key in default_run_kwargs:
            if key not in run_kwargs:
                run_kwargs[key] = default_run_kwargs[key]
            
        if self.verbose:
            print(f"Running dynesty with {sampler_kwargs['nlive']} live points on {dynesty_ncore} cores...")

        # Multi-run setup for minimum effective sample size
        all_samples = []
        all_weights = []
        all_logz = []
        all_run_times = []
        accumulated_samples = 0
        run_number = 1
        iMinSamples = max(min_ess, 1)

        while accumulated_samples < iMinSamples:
            if min_ess > 0:
                print(f"\nRun {run_number}: Need {max(0, iMinSamples - accumulated_samples)} more samples...")
                print("="*50)

            # Set timing for this run
            run_start_time = dynesty_t0 if run_number == 1 else time.time()

            # Pickle sampler?
            if save_iter is not None:
                run_sampler = True
                last_iter = 0
                while run_sampler == True:
                    dsampler.run_nested(maxiter=save_iter, **run_kwargs)
                    current_results = dsampler.results

                    file = os.path.join(self.savedir, f"dynesty_sampler_{self.like_fn_name}_run{run_number}.pkl")

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
                current_results = dsampler.results

            # Record current run statistics
            current_runtime = time.time() - run_start_time
            all_run_times.append(current_runtime)
            
            # Get samples and weights for this run
            current_samples = current_results.samples
            current_weights = np.exp(current_results.logwt - current_results.logz[-1])
            current_logz = current_results.logz[-1]
            
            # Resample weighted samples for this run
            current_resampled = dyfunc.resample_equal(current_samples, current_weights)
            all_samples.append(current_resampled)
            all_weights.append(current_weights)
            all_logz.append(current_logz)
            
            current_nsamples = current_resampled.shape[0]
            accumulated_samples += current_nsamples
            
            if min_ess > 0:
                print(f"Run {run_number} complete: {current_nsamples} samples, logZ = {current_logz:.3f}")
                print(f"Total accumulated samples: {accumulated_samples}")
            
            # If min_ess requirement met, break
            if accumulated_samples >= iMinSamples:
                break

            run_number += 1

            # Setup for next run
            if run_number <= 10:  # Limit to prevent infinite loops
                # Create a new sampler for the next run
                if mode == "dynamic":
                    dsampler = dynesty.DynamicNestedSampler(self.like_fn, 
                                                            self.prior_transform, 
                                                            ndim=self.ndim, 
                                                            **sampler_kwargs)
                else:
                    dsampler = dynesty.NestedSampler(self.like_fn, 
                                                     self.prior_transform, 
                                                     ndim=self.ndim, 
                                                     **sampler_kwargs)
            else:
                print(f"WARNING: Reached maximum of 10 runs, stopping with {accumulated_samples} samples")
                break
        
        # Combine results from all runs
        if len(all_samples) > 1:
            self.dynesty_samples = np.vstack(all_samples)
            # Use the best (highest) log evidence
            self.dynesty_logz = max(all_logz)
            print(f"\nCombined {len(all_samples)} runs into {self.dynesty_samples.shape[0]} total samples")
            print(f"Best log evidence: {self.dynesty_logz:.3f}")
        else:
            self.dynesty_samples = all_samples[0]
            self.dynesty_logz = all_logz[0]
        
        # Use final run for primary results
        self.dynesty_sampler = dsampler
        self.dynesty_results = current_results
        self.dynesty_logz_err = current_results.logzerr[-1]
        self.dynesty_runtime = sum(all_run_times)
        
        if self.like_fn_name == "true":
            self.dynesty_samples_true = self.dynesty_samples
        elif self.like_fn_name == "surrogate":
            self.dynesty_samples_surrogate = self.dynesty_samples  

        # record that dynesty has been run
        self.dynesty_run = True
        
        # record dynesty runtime
        self.dynesty_runtime = time.time() - dynesty_t0

        if self.cache:
            try:
                self.save()
            except:
                pass

        if samples_file is not None:
            fname = f"{self.savedir}/{samples_file}"
        else:
            if self.like_fn_name == "true":
                fname = f"{self.savedir}/dynesty_samples_final_{self.like_fn_name}.npz"
            else:
                if len(self.training_results["iteration"]) == 0:
                    current_iter = 0
                else:
                    current_iter = self.training_results["iteration"][-1]
                fname = f"{self.savedir}/dynesty_samples_final_{self.like_fn_name}_iter_{current_iter}.npz"
        print(f"Saved dynesty samples to {fname}")
        np.savez(fname, samples=self.dynesty_samples)


    def run_pymultinest(self, like_fn=None, prior_transform=None, sampler_kwargs={}, multi_proc=True, 
                        prior_transform_comment=None, samples_file=None, prefix=None, resume=False, 
                        n_clustering_params=None, outputfiles_basename=None, min_ess=int(1e4)):
        """
        Sample the posterior using the PyMultiNest nested sampling algorithm.
        
        This method uses the PyMultiNest package (Python wrapper for MultiNest) to perform 
        nested sampling on either the trained GP surrogate model or the true likelihood function.
        MultiNest is particularly effective for multi-modal posteriors and computing Bayesian 
        evidence with high accuracy.
        
        :param like_fn: (*callable, str, or None, optional*)
            Likelihood function to sample. Options:
            - None (default): Uses the trained GP surrogate model (self.surrogate_log_likelihood)
            - "surrogate", "gp": Uses the GP surrogate model explicitly
            - "true": Uses the true likelihood function (self.true_log_likelihood)
            - callable: Custom likelihood function with signature like_fn(theta)
            Default is None.
            
        :param prior_transform: (*callable or None, optional*)
            Prior transformation function that maps from unit hypercube [0,1]^ndim
            to the parameter space. Should have signature prior_transform(cube) where cube
            is array of shape (ndim,) with values in [0,1]. The function should modify
            cube in-place. If None, uses uniform prior with bounds from self.bounds.
            Default is None.
            
        :param sampler_kwargs: (*dict, optional*)
            Additional keyword arguments passed to pymultinest.run().
            Common options include:
            - 'n_live_points': Number of live points (default: 1000)
            - 'evidence_tolerance': Target evidence uncertainty (default: 0.5)
            - 'sampling_efficiency': Sampling efficiency parameter (default: 0.8)
            - 'n_iter_before_update': Iterations before evidence/posterior update (default: 100)
            - 'null_log_evidence': Null evidence for model comparison (default: -1e90)
            - 'max_modes': Maximum number of modes to find (default: 100)
            - 'mode_tolerance': Mode separation tolerance (default: -1e90)
            - 'seed': Random seed for reproducibility (default: -1, auto)
            - 'verbose': Verbosity level (default: True)
            - 'importance_nested_sampling': Use importance nested sampling (default: True)
            - 'multimodal': Enable multimodal mode detection (default: True)
            - 'const_efficiency_mode': Use constant efficiency mode (default: False)
            Default is {}.
            
        :param multi_proc: (*bool, optional*)
            Whether to use multiprocessing. If True, uses self.ncore processes.
            MultiNest handles parallelization internally when MPI is available.
            
            .. note::
                This parameter is ignored for PyMultiNest as it uses MPI for 
                parallelization, not Python's multiprocessing. When PyMultiNest
                runs with MPI, other alabi functions automatically disable their
                multiprocessing pools to avoid conflicts.
                
            Default is True.
            
        :param prior_transform_comment: (*str or None, optional*)
            Comment describing the prior function for logging purposes. If None
            and prior_transform is provided, attempts to extract function name.
            Default is None.
            
        :param samples_file: (*str or None, optional*)
            If provided, saves the final samples to this file in NumPy .npz format.
            Default is None.
            
        :param prefix: (*str or None, optional*)
            Prefix for MultiNest output files. If None, uses default based on
            likelihood function name and current directory. Default is None.
            
        :param resume: (*bool, optional*)
            Whether to resume from previous run if output files exist.
            Default is False.
            
        :param n_clustering_params: (*int or None, optional*)
            Number of parameters to use for mode clustering. If None, uses all
            parameters (ndim). Set to lower value if some parameters are nuisance.
            Default is None.
            
        :param outputfiles_basename: (*str or None, optional*)
            Base name for MultiNest output files. If None, constructs from savedir
            and likelihood function name. Default is None.
            
        :param min_ess: (*int, optional*)
            Minimum effective sample size. If the number of final samples is less than 
            min_ess, will run additional sampling rounds and combine samples until the 
            total number of samples exceeds min_ess. Default is 0 (no minimum required).
            
        Attributes Set
        --------------
        pymultinest_samples : ndarray of shape (nsamples, ndim)
            Posterior samples from MultiNest
        pymultinest_samples_true : ndarray of shape (nsamples, ndim)
            Posterior samples when using true likelihood (like_fn="true")
        pymultinest_samples_surrogate : ndarray of shape (nsamples, ndim)
            Posterior samples when using surrogate likelihood
        pymultinest_weights : ndarray of shape (nsamples,)
            Sample weights from nested sampling
        pymultinest_logz : float
            Log Bayesian evidence estimate
        pymultinest_logz_err : float
            Uncertainty in log evidence estimate
        pymultinest_run : bool
            Flag indicating PyMultiNest has been successfully run
        pymultinest_runtime : float
            Wall-clock time taken for MultiNest sampling in seconds
        pymultinest_analyzer : pymultinest.Analyzer
            MultiNest analyzer object for accessing detailed results
        like_fn_name : str
            Name of likelihood function used ("true", "surrogate", or "custom")
        prior_transform_comment : str
            Description of prior function used
            
        .. note::
            PyMultiNest is particularly well-suited for:
            
            - Multi-modal posterior exploration with automatic mode detection
            - High-accuracy Bayesian evidence computation for model comparison
            - Robust sampling without manual tuning of MCMC parameters
            - Handling complex, irregular posterior shapes
            
            MultiNest generates several output files including detailed posterior
            samples, evidence estimates, and mode information. These files are
            saved to the model's savedir for later analysis.
            
            **MPI and Multiprocessing Compatibility:**
            
            PyMultiNest uses MPI for parallelization across multiple nodes/cores.
            When MPI is active, alabi automatically disables Python multiprocessing
            in other functions (run_emcee, run_dynesty) to prevent conflicts.
            This ensures that:
            
            - PyMultiNest can run efficiently with MPI
            - Other alabi functions fall back to serial execution when MPI is detected
            - No deadlocks or resource conflicts occur between MPI and multiprocessing
            
            To run PyMultiNest with MPI:
            
            >>> # Single node, multiple cores
            >>> sm.run_pymultinest()  # Uses OpenMP if available
            
            >>> # Multiple nodes with MPI (run from command line)
            >>> # mpirun -n 4 python your_script.py
            
        :example:
            Sample surrogate model with default settings:
            
            >>> sm.run_pymultinest()
            
            Sample true likelihood with more live points:
            
            >>> sm.run_pymultinest(like_fn="true", 
            ...                   sampler_kwargs={'n_live_points': 2000})
            
            Use custom prior with bounds [-10, 10] for each parameter:
            
            >>> def my_prior(cube):
            ...     for i in range(len(cube)):
            ...         cube[i] = 20*cube[i] - 10  # maps [0,1] to [-10,10]
            >>> sm.run_pymultinest(prior_transform=my_prior)
            
            Enable multimodal mode detection with high accuracy:
            
            >>> sm.run_pymultinest(sampler_kwargs={
            ...     'multimodal': True,
            ...     'evidence_tolerance': 0.1,
            ...     'max_modes': 20
            ... })
            
            Run with custom output file prefix and resume capability:
            
            >>> sm.run_pymultinest(prefix="my_run_", resume=True)
            
        References
        ----------
        PyMultiNest documentation: https://johannesbuchner.github.io/PyMultiNest/
        Feroz et al. (2009): "MultiNest: an efficient and robust Bayesian inference
        tool for cosmology and particle physics", MNRAS, 398, 1601-1614
        """
        
        try:
            import pymultinest
        except ImportError:
            raise ImportError("PyMultiNest is required but not installed. "
                            "Install with: pip install pymultinest")
        
        # Start timing
        pymultinest_t0 = time.time()
        
        # Determine likelihood function and name
        if like_fn is None:
            print("Initializing PyMultiNest with self.surrogate_log_likelihood surrogate model as likelihood.")
            self.like_fn_name = "surrogate"
            self.like_fn = self.surrogate_log_likelihood
        elif callable(like_fn):
            # like_fn is a function/callable
            if like_fn == self.surrogate_log_likelihood:
                print("Initializing PyMultiNest with self.surrogate_log_likelihood surrogate model as likelihood.")
                self.like_fn_name = "surrogate"
                self.like_fn = self.surrogate_log_likelihood
            elif like_fn == self.true_log_likelihood:
                print("Initializing PyMultiNest with self.true_log_likelihood as likelihood.")
                self.like_fn_name = "true"
                self.like_fn = like_fn
            else:
                # Custom function provided
                print("Initializing PyMultiNest with user-provided likelihood function.")
                self.like_fn_name = "custom"
                self.like_fn = like_fn
        elif isinstance(like_fn, str):
            if like_fn.lower() in ["surrogate", "gp"]:
                print("Initializing PyMultiNest with self.surrogate_log_likelihood surrogate model as likelihood.")
                self.like_fn_name = "surrogate"
                self.like_fn = self.surrogate_log_likelihood
            elif like_fn.lower() == "true":
                print("Initializing PyMultiNest with self.true_log_likelihood as likelihood.")
                self.like_fn_name = "true"
                self.like_fn = self.true_log_likelihood
            else:
                raise ValueError(f"Invalid like_fn string: {like_fn}. "
                               "Must be 'surrogate', 'gp', 'true', or callable.")
        else:
            raise ValueError(f"like_fn must be callable, string, or None. Got {type(like_fn)}")
        
        # Set up prior transformation function
        if prior_transform is None:
            # Default uniform prior using self.bounds
            prior_transform_fn = partial(ut.prior_transform_uniform, bounds=self.bounds)
            self.prior_transform_comment = f"Default uniform prior transform.\nPrior function: ut.prior_transform_uniform\nBounds: {self.bounds}"
        else:
            prior_transform_fn = prior_transform
            if prior_transform_comment is None:
                if hasattr(prior_transform, '__name__'):
                    self.prior_transform_comment = f"Custom prior function: {prior_transform.__name__}"
                else:
                    self.prior_transform_comment = "Custom prior function"
            else:
                self.prior_transform_comment = prior_transform_comment
        
        # Create PyMultiNest-compatible wrapper for prior transform
        # PyMultiNest expects Prior(cube, ndim, nparams) where cube is modified in-place
        def pymultinest_prior_wrapper(cube, ndim, nparams):
            """Wrapper to make prior transform compatible with PyMultiNest calling convention."""
            # Extract values from ctypes array to regular Python list
            cube_values = [cube[i] for i in range(ndim)]
            cube_array = np.array(cube_values)
            
            # Apply the prior transformation
            transformed = prior_transform_fn(cube_array)
            
            # Modify cube in-place as expected by PyMultiNest
            for i in range(ndim):
                cube[i] = transformed[i]
        
        # Use the wrapper as the actual prior function
        prior_transform = pymultinest_prior_wrapper
        
        # Create PyMultiNest-compatible wrapper for likelihood function  
        # PyMultiNest expects LogLikelihood(cube, ndim, nparams) where cube contains parameters
        def pymultinest_likelihood_wrapper(cube, ndim, nparams):
            """Wrapper to make likelihood function compatible with PyMultiNest calling convention."""
            # Extract values from ctypes array to regular Python list
            params_values = [cube[i] for i in range(ndim)]
            params = np.array(params_values)
            return self.like_fn(params)
        
        # Use the wrapper as the actual likelihood function
        likelihood_fn = pymultinest_likelihood_wrapper
        
        # Set up output file basename
        if outputfiles_basename is None:
            if prefix is None:
                if len(self.training_results.get("iteration", [])) == 0:
                    current_iter = 0
                else:
                    current_iter = self.training_results["iteration"][-1]
                prefix = f"pymultinest_{self.like_fn_name}_iter_{current_iter}_"
            outputfiles_basename = f"{self.savedir}/{prefix}"
        
        # Set default sampler kwargs
        default_sampler_kwargs = {
            'n_live_points': 1000,
            'evidence_tolerance': 0.5,
            'sampling_efficiency': 0.8,
            'n_iter_before_update': 100,
            'null_log_evidence': -1e90,
            'max_modes': 100,
            'mode_tolerance': -1e90,
            'seed': -1,
            'verbose': True,
            'importance_nested_sampling': True,
            'multimodal': True,
            'const_efficiency_mode': False,
        }
        
        # Update with user-provided kwargs
        final_sampler_kwargs = {**default_sampler_kwargs, **sampler_kwargs}
        
        # Set clustering parameters
        if n_clustering_params is None:
            n_clustering_params = self.ndim
        
        # Multi-run setup for minimum effective sample size
        all_samples = []
        all_weights = []
        all_logz = []
        all_logz_err = []
        all_run_times = []
        accumulated_samples = 0
        run_number = 1
        
        while accumulated_samples < max(min_ess, 1):
            if min_ess > 0:
                print(f"\nRun {run_number}: Need {max(0, min_ess - accumulated_samples)} more samples...")
                print("="*50)
            
            # Create unique output basename for this run
            current_outputfiles_basename = f"{outputfiles_basename}_run{run_number:02d}" if min_ess > 0 else outputfiles_basename
            
            print(f"Running PyMultiNest with {final_sampler_kwargs['n_live_points']} live points...")
            print(f"Evidence tolerance: {final_sampler_kwargs['evidence_tolerance']}")
            print(f"Output files: {current_outputfiles_basename}*")
            print(f"Prior: {self.prior_transform_comment}")
            
            # Start timing this run
            current_pymultinest_t0 = time.time()
            
            # Run MultiNest
            pymultinest.run(
                LogLikelihood=likelihood_fn,
                Prior=prior_transform,
                n_dims=self.ndim,
                n_params=self.ndim,
                n_clustering_params=n_clustering_params,
                outputfiles_basename=current_outputfiles_basename,
                resume=resume,
                **final_sampler_kwargs
            )
            
            # Fix malformed scientific notation in PyMultiNest output files
            self._fix_pymultinest_output_format(current_outputfiles_basename)
            
            # Analyze results for this run
            current_analyzer = pymultinest.Analyzer(
                outputfiles_basename=current_outputfiles_basename,
                n_params=self.ndim
            )
            
            # Get samples and evidence for this run
            current_samples = current_analyzer.get_equal_weighted_posterior()
            current_samples_only = current_samples[:, :-1]  # Remove log-likelihood column
            current_weights = np.ones(len(current_samples_only))  # Equal weights
            
            # Get evidence estimates for this run
            current_stats = current_analyzer.get_stats()
            current_logz = current_stats['global evidence']
            current_logz_err = current_stats['global evidence error']
            
            # Record this run's results
            all_samples.append(current_samples_only)
            all_weights.append(current_weights)
            all_logz.append(current_logz)
            all_logz_err.append(current_logz_err)
            
            current_nsamples = len(current_samples_only)
            accumulated_samples += current_nsamples
            
            # Record runtime for this run
            current_run_time = time.time() - current_pymultinest_t0
            all_run_times.append(current_run_time)
            
            if min_ess > 0:
                print(f"Run {run_number} complete: {current_nsamples} samples, logZ = {current_logz:.3f} ± {current_logz_err:.3f}")
                print(f"Total accumulated samples: {accumulated_samples}")
            
            # If min_ess requirement met, break
            if accumulated_samples >= min_ess:
                break
                
            run_number += 1
            
            # Limit to prevent infinite loops
            if run_number > 10:
                print(f"WARNING: Reached maximum of 10 runs, stopping with {accumulated_samples} samples")
                break
        
        # Combine results from all runs
        if len(all_samples) > 1:
            self.pymultinest_samples = np.vstack(all_samples)
            self.pymultinest_weights = np.concatenate(all_weights)
            # Use weighted average of log evidence (weights by number of samples)
            sample_counts = np.array([len(s) for s in all_samples])
            total_samples = np.sum(sample_counts)
            weights = sample_counts / total_samples
            self.pymultinest_logz = np.average(all_logz, weights=weights)
            self.pymultinest_logz_err = np.sqrt(np.average(np.array(all_logz_err)**2, weights=weights))
            print(f"\nCombined {len(all_samples)} runs into {len(self.pymultinest_samples)} total samples")
            print(f"Combined log evidence: {self.pymultinest_logz:.3f} ± {self.pymultinest_logz_err:.3f}")
        else:
            self.pymultinest_samples = all_samples[0]
            self.pymultinest_weights = all_weights[0]
            self.pymultinest_logz = all_logz[0]
            self.pymultinest_logz_err = all_logz_err[0]
        
        # Set the final analyzer to the last run for compatibility
        self.pymultinest_analyzer = current_analyzer
        
        # Calculate total runtime
        self.pymultinest_runtime = sum(all_run_times)
        
        # Store samples based on likelihood function used
        if self.like_fn_name == "true":
            self.pymultinest_samples_true = self.pymultinest_samples.copy()
            print(f"PyMultiNest complete. {len(self.pymultinest_samples_true)} posterior samples collected.")
        else:
            self.pymultinest_samples_surrogate = self.pymultinest_samples.copy()
            print(f"PyMultiNest complete. {len(self.pymultinest_samples_surrogate)} posterior samples collected.")
        
        print(f"Log Evidence: {self.pymultinest_logz:.3f} ± {self.pymultinest_logz_err:.3f}")
        
        # Record that PyMultiNest has been run
        self.pymultinest_run = True
        
        print(f"PyMultiNest runtime: {self.pymultinest_runtime:.2f} seconds")
        
        # Save results if caching is enabled
        if self.cache:
            try:
                self.save()
            except:
                pass
            
        # Save samples to file
        if samples_file is not None:
            fname = f"{self.savedir}/{samples_file}"
        else:
            if self.like_fn_name == "true":
                fname = f"{self.savedir}/pymultinest_samples_final_{self.like_fn_name}.npz"
            else:
                if len(self.training_results.get("iteration", [])) == 0:
                    current_iter = 0
                else:
                    current_iter = self.training_results["iteration"][-1]
                fname = f"{self.savedir}/pymultinest_samples_final_{self.like_fn_name}_iter_{current_iter}.npz"
        
        print(f"Saved PyMultiNest samples to {fname}")
        np.savez(fname, 
                samples=self.pymultinest_samples,
                weights=self.pymultinest_weights,
                logz=self.pymultinest_logz,
                logz_err=self.pymultinest_logz_err)


    def run_ultranest(self, like_fn=None, prior_transform=None, sampler_kwargs={}, run_kwargs={},
                      multi_proc=False, prior_transform_comment=None, samples_file=None,
                      log_dir=None, resume="overwrite", min_ess=int(1e4), slice_steps=0):
        """
        Sample the posterior using the UltraNest nested sampling algorithm.
        
        This method uses the UltraNest package to perform nested sampling on either
        the trained GP surrogate model or the true likelihood function. UltraNest
        is a highly robust nested sampling algorithm that automatically adapts to
        the problem complexity and provides reliable evidence computation.
        
        :param like_fn: (*callable, str, or None, optional*)
            Likelihood function to sample. Options:
            - None (default): Uses the trained GP surrogate model (self.surrogate_log_likelihood)
            - "surrogate", "gp": Uses the GP surrogate model explicitly
            - "true": Uses the true likelihood function (self.true_log_likelihood)
            - callable: Custom likelihood function with signature like_fn(theta)
            Default is None.
            
        :param prior_transform: (*callable or None, optional*)
            Prior transformation function that maps from unit hypercube [0,1]^ndim
            to the parameter space. Should have signature prior_transform(cube) where
            cube is array of shape (ndim,) with values in [0,1]. Must return transformed
            parameters as array. If None, creates uniform prior from self.bounds.
            Default is None.
            
        :param sampler_kwargs: (*dict, optional*)
            Additional keyword arguments passed to UltraNest ReactiveNestedSampler().
            Common options include:
            - 'derived_param_names': List of derived parameter names (default: [])
            - 'wrapped_params': List of bool indicating circular parameters (default: None)
            - 'resume': Resume behavior 'resume'/'overwrite'/'subfolder' (default: 'subfolder')
            - 'run_num': Run number for subdirectory creation (default: None)
            - 'num_test_samples': Number of test samples for validation (default: 2)
            - 'draw_multiple': Enable dynamic point drawing (default: True)  
            - 'num_bootstraps': Number of bootstrap samples (default: 30)
            - 'vectorized': Whether functions accept arrays (default: False)
            - 'ndraw_min': Minimum points to draw per iteration (default: 128)
            - 'ndraw_max': Maximum points to draw per iteration (default: 65536)
            - 'storage_backend': Storage format 'hdf5'/'csv'/'tsv' (default: 'hdf5')
            - 'warmstart_max_tau': Warmstart maximum tau (default: -1)
            Default is {}.
            
        :param run_kwargs: (*dict, optional*)
            Additional keyword arguments passed to sampler.run().
            Common options include:
            - 'update_interval_volume_fraction': Volume fraction for region updates (default: 0.8)
            - 'update_interval_ncall': Number of calls between updates (optional, omit for auto)
            - 'log_interval': Iterations between status updates (optional, omit for auto)
            - 'show_status': Show integration progress (default: True)
            - 'viz_callback': Visualization callback function (default: False, disabled)
            - 'dlogz': Target log-evidence uncertainty (default: 0.5)
            - 'dKL': Target posterior uncertainty in nats (default: 0.5)
            - 'frac_remain': Fraction of evidence remaining to terminate (default: 0.01)
            - 'Lepsilon': Likelihood contour tolerance (default: 0.001)
            - 'min_ess': Target effective sample size (default: 400)
            - 'max_iters': Maximum number of iterations (optional, omit for unlimited)
            - 'max_ncalls': Maximum number of likelihood calls (optional, omit for unlimited)
            - 'max_num_improvement_loops': Maximum improvement loops (default: -1)
            - 'min_num_live_points': Minimum number of live points (default: 400)
            - 'cluster_num_live_points': Live points per cluster (default: 40)
            - 'insertion_test_zscore_threshold': Z-score threshold for insertion test (default: 4)
            - 'insertion_test_window': Window size for insertion test (default: 10)
            - 'region_class': Region sampling class (optional, can be passed via kwargs)
            - 'widen_before_initial_plateau_num_warn': Warning threshold for plateau (default: 10000)
            - 'widen_before_initial_plateau_num_max': Maximum plateau points (optional, omit for auto)
            Default is {}.
            
        :param multi_proc: (*bool, optional*)
            **Deprecated and ignored.** This parameter is kept for backwards compatibility
            but no longer has any effect. UltraNest now runs in MPI-compatible mode without
            multiprocessing pools to avoid conflicts with MPI environments.
            Default is False.
            
        :param prior_transform_comment: (*str or None, optional*)
            Comment describing the prior transform for logging purposes. If None
            and prior_transform is provided, attempts to extract function name.
            Default is None.
            
        :param samples_file: (*str or None, optional*)
            If provided, saves the final samples to this file in NumPy .npz format.
            Default is None.
            
        :param log_dir: (*str or None, optional*)
            Directory to store UltraNest output files and logs. If None, uses
            a subdirectory in self.savedir. Default is None.
            
        :param resume: (*str, optional*)
            Resume behavior for interrupted runs. Options:
            - 'resume': Resume if possible, otherwise start fresh
            - 'resume-similar': Resume with similar but not identical setup
            - 'overwrite': Always start fresh, overwriting existing files (default)
            - 'subfolder': Create new timestamped subfolder
            Default is 'overwrite'.
            
        :param min_ess: (*int, optional*)
            Minimum effective sample size. If the number of final samples is less than 
            min_ess, will run additional sampling rounds and combine samples until the 
            total number of samples exceeds min_ess. Default is 0 (no minimum required).
            
        Attributes Set
        --------------
        ultranest_results : ultranest.integrator.Result
            Complete UltraNest results object with samples, evidence, etc.
        ultranest_samples : ndarray of shape (nsamples, ndim)
            Equally weighted posterior samples from UltraNest
        ultranest_samples_true : ndarray of shape (nsamples, ndim)
            Posterior samples when using true likelihood (like_fn="true")
        ultranest_samples_surrogate : ndarray of shape (nsamples, ndim)
            Posterior samples when using surrogate likelihood
        ultranest_weights : ndarray of shape (nsamples,)
            Sample weights (typically all equal after resampling)
        ultranest_logz : float
            Log Bayesian evidence estimate
        ultranest_logz_err : float
            Uncertainty in log evidence estimate
        ultranest_run : bool
            Flag indicating UltraNest has been successfully run
        ultranest_runtime : float
            Wall-clock time taken for UltraNest sampling in seconds
        ultranest_sampler : ultranest.ReactiveNestedSampler
            UltraNest sampler object for accessing detailed information
        like_fn_name : str
            Name of likelihood function used ("true", "surrogate", or "custom")
        prior_transform_comment : str
            Description of prior transform used
            
        .. note::
            UltraNest is particularly well-suited for:
            
            - Robust nested sampling without manual tuning
            - Automatic adaptation to problem complexity
            - High-dimensional and multi-modal problems
            - Reliable evidence computation for model comparison
            - Problems with complex, irregular likelihood shapes
            - MPI environments (runs in serial mode to avoid multiprocessing conflicts)
            
            UltraNest automatically determines the number of live points and
            adapts its sampling strategy based on the problem characteristics.
            It provides excellent performance across a wide range of problems
            without requiring parameter tuning.
            
            **MPI Compatibility:** This function runs UltraNest in serial mode,
            making it fully compatible with MPI environments. While UltraNest
            itself can use MPI for parallelization, this implementation avoids
            multiprocessing pools that can conflict with MPI.
            
        :example:
            Sample surrogate model with default settings:
            
            >>> sm.run_ultranest()
            
            Sample true likelihood with custom termination criteria:
            
            >>> sm.run_ultranest(like_fn="true", 
            ...                  run_kwargs={'dlogz': 0.1, 'min_ess': 1000})
            
            Use custom prior transform with bounds [-5, 5] for each parameter:
            
            >>> def my_prior_transform(cube):
            ...     return 10 * cube - 5  # maps [0,1] to [-5,5]
            >>> sm.run_ultranest(prior_transform=my_prior_transform)
            
            Run with increased live points for better accuracy:
            
            >>> sm.run_ultranest(like_fn="true",
            ...                  run_kwargs={'min_num_live_points': 800})
            
            Customize output directory and resume behavior:
            
            >>> sm.run_ultranest(log_dir="ultranest_output", 
            ...                  resume="overwrite")
            
        References
        ----------
        UltraNest documentation: https://johannesbuchner.github.io/UltraNest/
        Buchner (2021): "UltraNest - a robust, general purpose Bayesian inference
        library for cosmology and particle physics", Journal of Open Source Software
        """
        
        try:
            import ultranest
            from ultranest import ReactiveNestedSampler
        except ImportError:
            raise ImportError("UltraNest is required but not installed. "
                            "Install with: pip install ultranest")
        
        # Start timing
        ultranest_t0 = time.time()
        
        # Determine likelihood function and name
        if like_fn is None:
            print("Initializing UltraNest with self.surrogate_log_likelihood surrogate model as likelihood.")
            self.like_fn_name = "surrogate"
            self.like_fn = self.surrogate_log_likelihood
        elif callable(like_fn):
            # like_fn is a function/callable
            if like_fn == self.surrogate_log_likelihood:
                print("Initializing UltraNest with self.surrogate_log_likelihood surrogate model as likelihood.")
                self.like_fn_name = "surrogate"
                self.like_fn = self.surrogate_log_likelihood
            elif like_fn == self.true_log_likelihood:
                print("Initializing UltraNest with self.true_log_likelihood as likelihood.")
                self.like_fn_name = "true"
                self.like_fn = like_fn
            else:
                # Custom function provided
                print("Initializing UltraNest with user-provided likelihood function.")
                self.like_fn_name = "custom"
                self.like_fn = like_fn
        elif isinstance(like_fn, str):
            if like_fn.lower() in ["surrogate", "gp"]:
                print("Initializing UltraNest with self.surrogate_log_likelihood surrogate model as likelihood.")
                self.like_fn_name = "surrogate"
                self.like_fn = self.surrogate_log_likelihood
            elif like_fn.lower() == "true":
                print("Initializing UltraNest with self.true_log_likelihood as likelihood.")
                self.like_fn_name = "true"
                self.like_fn = self.true_log_likelihood
            else:
                raise ValueError(f"Invalid like_fn string: {like_fn}. "
                               "Must be 'surrogate', 'gp', 'true', or callable.")
        else:
            raise ValueError(f"like_fn must be callable, string, or None. Got {type(like_fn)}")
        
        # Set up prior transformation function
        if prior_transform is None:
            # Default uniform prior using self.bounds
            prior_transform_fn = partial(ut.prior_transform_uniform, bounds=self.bounds)
            self.prior_transform_comment = f"Uniform prior with bounds {self.bounds}"
        else:
            prior_transform_fn = prior_transform
            if prior_transform_comment is None:
                if hasattr(prior_transform, '__name__'):
                    self.prior_transform_comment = f"Custom prior transform: {prior_transform.__name__}"
                else:
                    self.prior_transform_comment = "Custom prior transform"
            else:
                self.prior_transform_comment = prior_transform_comment
        
        # Set up log directory
        if log_dir is None:
            if len(self.training_results.get("iteration", [])) == 0:
                current_iter = 0
            else:
                current_iter = self.training_results["iteration"][-1]
            log_dir = f"{self.savedir}/ultranest_{self.like_fn_name}_iter_{current_iter}"
        
        # Set default sampler kwargs (these go to ReactiveNestedSampler constructor)
        default_sampler_kwargs = {
            'derived_param_names': [],
            'wrapped_params': None,
            'num_test_samples': 2,
            'draw_multiple': True,
            'num_bootstraps': 30,
            'vectorized': False,
            'ndraw_min': 128,
            'ndraw_max': 65536,
            'storage_backend': 'hdf5',
            'warmstart_max_tau': -1,
        }
        
        # Update with user-provided kwargs
        final_sampler_kwargs = {**default_sampler_kwargs, **sampler_kwargs}
        
        # Set default run kwargs (these go to sampler.run() method)
        default_run_kwargs = {
            'update_interval_volume_fraction': 0.8,
            'show_status': True,
            'viz_callback': False,  # Disable visualization to avoid ipywidgets dependency
            'dlogz': 0.5,
            'dKL': 0.5,
            'frac_remain': 0.01,
            'Lepsilon': 0.001,
            'min_ess': 400,
            'max_num_improvement_loops': 1,
            'min_num_live_points': 400,
            'cluster_num_live_points': 40,
            'insertion_test_zscore_threshold': 4,
            'insertion_test_window': 10,
            'widen_before_initial_plateau_num_warn': 10000,
        }
        
        # Update with user-provided kwargs
        final_run_kwargs = {**default_run_kwargs, **run_kwargs}
        
        # Check if MPI is active for informational purposes
        if parallel_utils.is_mpi_active():
            print("MPI environment detected. UltraNest will run in MPI-compatible mode.")
        elif multi_proc:
            print("Warning: multi_proc=True ignored. UltraNest now runs in MPI-compatible serial mode.")
        
        # Define wrapped likelihood function for UltraNest
        def ultranest_likelihood(params):
            """Likelihood function wrapper for UltraNest."""
            return self.like_fn(params)
        
        print(f"Running UltraNest with {final_run_kwargs['min_num_live_points']} minimum live points...")
        print(f"Log directory: {log_dir}")
        print(f"Execution mode: MPI-compatible (no multiprocessing pools)")
        print(f"Prior: {self.prior_transform_comment}")
        print(f"Termination: dlogz={final_run_kwargs['dlogz']}, min_ess={final_run_kwargs['min_ess']}\n")

        # Multi-run setup for minimum effective sample size
        all_samples = []
        all_weights = []
        all_logz = []
        all_logz_err = []
        all_run_times = []
        accumulated_samples = 0
        run_number = 1
        param_names = [f"param_{i}" for i in range(self.ndim)]
        
        while accumulated_samples < min_ess:
            if min_ess > 0:
                print(f"\nRun {run_number}: Need {max(0, min_ess - accumulated_samples)} more samples...")
                print("="*50)
            
            # Set timing for this run
            run_start_time = ultranest_t0 if run_number == 1 else time.time()
            
            # Create a new sampler for each run (with updated log_dir if needed)
            current_log_dir = log_dir if run_number == 1 else f"{log_dir}_run{run_number}" if log_dir else None
            
            # Create UltraNest sampler instance
            self.ultranest_sampler = ReactiveNestedSampler(
                param_names=param_names,
                loglike=ultranest_likelihood,
                transform=prior_transform_fn,
                log_dir=current_log_dir,
                resume=resume if run_number == 1 else "overwrite",  # Only resume on first run
                **final_sampler_kwargs
            )
            
            if slice_steps > 0:
                from ultranest import stepsampler
                self.ultranest_sampler.stepsampler = stepsampler.SliceSampler(
                    nsteps=slice_steps,
                    generate_direction=stepsampler.generate_mixture_random_direction,
                )
            
            # Run UltraNest sampling (always in serial/MPI-compatible mode)
            current_results = self.ultranest_sampler.run(
                **final_run_kwargs
            )
            
            # Record current run statistics
            current_runtime = time.time() - run_start_time
            all_run_times.append(current_runtime)
            
            # Extract results for this run
            current_samples = current_results['samples']
            
            # Extract weights from weighted_samples if available, otherwise use equal weights
            if 'weighted_samples' in current_results:
                current_weights = current_results['weighted_samples']['weights']
            else:
                # For equally weighted samples, create uniform weights
                current_weights = np.ones(len(current_samples)) / len(current_samples)
            
            current_logz = current_results['logz']
            current_logz_err = current_results['logzerr']
            
            # Store results from this run
            all_samples.append(current_samples)
            all_weights.append(current_weights)
            all_logz.append(current_logz)
            all_logz_err.append(current_logz_err)
            
            current_nsamples = len(current_samples)
            accumulated_samples += current_nsamples
            
            if min_ess > 0:
                print(f"Run {run_number} complete: {current_nsamples} samples, logZ = {current_logz:.3f} ± {current_logz_err:.3f}")
                print(f"Total accumulated samples: {accumulated_samples}")
            
            # If min_ess requirement met, break
            if accumulated_samples >= min_ess:
                break
                
            run_number += 1
            
            # Limit to prevent infinite loops
            if run_number > 10:
                print(f"WARNING: Reached maximum of 10 runs, stopping with {accumulated_samples} samples")
                break
        
        # Combine results from all runs
        if len(all_samples) > 1:
            self.ultranest_samples = np.vstack(all_samples)
            self.ultranest_weights = np.concatenate(all_weights)
            # Use the best (highest) log evidence
            best_run_idx = np.argmax(all_logz)
            self.ultranest_logz = all_logz[best_run_idx]
            self.ultranest_logz_err = all_logz_err[best_run_idx]
            print(f"\nCombined {len(all_samples)} runs into {len(self.ultranest_samples)} total samples")
            print(f"Best log evidence: {self.ultranest_logz:.3f} ± {self.ultranest_logz_err:.3f}")
        else:
            self.ultranest_samples = all_samples[0]
            self.ultranest_weights = all_weights[0]
            self.ultranest_logz = all_logz[0]
            self.ultranest_logz_err = all_logz_err[0]
        
        # Use final run for primary results
        self.ultranest_results = current_results
        self.ultranest_runtime = sum(all_run_times)
        
        # Store samples based on likelihood function used
        if self.like_fn_name == "true":
            self.ultranest_samples_true = self.ultranest_samples.copy()
            print(f"UltraNest complete. {len(self.ultranest_samples_true)} posterior samples collected.")
        else:
            self.ultranest_samples_surrogate = self.ultranest_samples.copy()
            print(f"UltraNest complete. {len(self.ultranest_samples_surrogate)} posterior samples collected.")
        
        print(f"Log Evidence: {self.ultranest_logz:.3f} ± {self.ultranest_logz_err:.3f}")
        print(f"Effective Sample Size: {self.ultranest_results.get('ess', 'N/A')}")
        print(f"Number of likelihood evaluations: {self.ultranest_results.get('ncall', 'N/A')}")
        
        # Record that UltraNest has been run
        self.ultranest_run = True
        
        print(f"UltraNest runtime: {self.ultranest_runtime:.2f} seconds\n")
        
        # Save results if caching is enabled
        if self.cache:
            try:
                self.save()
            except:
                pass
            
        # Save samples to file
        if samples_file is not None:
            fname = f"{self.savedir}/{samples_file}"
        else:
            if self.like_fn_name == "true":
                fname = f"{self.savedir}/ultranest_samples_final_true.npz"
            else:
                if len(self.training_results.get("iteration", [])) == 0:
                    current_iter = 0
                else:
                    current_iter = self.training_results["iteration"][-1]
                fname = f"{self.savedir}/ultranest_samples_final_surrogate_iter_{current_iter}.npz"
        
        print(f"Saved UltraNest samples to {fname}")
        np.savez(fname, 
                samples=self.ultranest_samples,
                weights=self.ultranest_weights,
                logz=self.ultranest_logz,
                logz_err=self.ultranest_logz_err)


    def plot(self, plots=None, show=False, cb_rng=[None, None], log_scale=False):
        """
        Generate diagnostic plots for training progress, GP performance, and MCMC results.
        
        This method creates various diagnostic plots to assess the quality of the surrogate
        model training, GP hyperparameter optimization, and MCMC sampling results. Plots
        are automatically saved to the model's save directory.
        
        :param plots: (*list of str, optional*)
            List of plot types to generate. Each plot requires specific data to be available
            (e.g., 'emcee_corner' requires run_emcee() to have been called first). If None,
            no plots are generated. Available options:
            
            **Training diagnostics:**
            - 'test_mse': Mean squared error vs training iteration
            - 'test_scaled_mse': Scaled MSE vs training iteration  
            - 'test_log_mse': Log-scale MSE vs training iteration
            - 'gp_hyperparameters': GP hyperparameter evolution during training
            - 'gp_train_time': GP training time vs iteration
            - 'gp_train_corner': Corner plot of final training samples
            - 'gp_train_scatter': Scatter plot of training samples vs predictions
            
            **GP visualization (2D only):**
            - 'gp_fit_2D': 2D contour plot of GP surrogate surface
            
            **MCMC diagnostics:**
            - 'emcee_corner': Corner plot of emcee posterior samples
            - 'emcee_walkers': Walker trajectories for emcee chains
            - 'dynesty_corner': Corner plot of dynesty posterior samples  
            - 'dynesty_corner_kde': KDE version of dynesty corner plot
            - 'dynesty_traceplot': Trace plot of dynesty sampling
            - 'dynesty_runplot': Dynesty convergence diagnostics
            
            **Comparison plots:**
            - 'mcmc_comparison': Compare emcee and dynesty posteriors
            
            **Convenience options:**
            - 'gp_all': Generate all available GP training plots
            
            Default is None.
            
        :param show: (*bool, optional*)
            Whether to display plots interactively in addition to saving them.
            If False, plots are only saved to disk. Default is False.
            
        :param cb_rng: (*list of [float, float], optional*)
            Colorbar range for 2D contour plots as [vmin, vmax]. If [None, None],
            uses automatic range determination. Only applies to plots with colorbars
            like 'gp_fit_2D'. Default is [None, None].
            
        :param log_scale: (*bool, optional*)
            Whether to use logarithmic color scale for 2D contour plots. If True,
            applies matplotlib.colors.LogNorm to the colorbar. Only applies to
            plots with colorbars. Default is False.
            
        :returns: *None or matplotlib.figure.Figure*
            Some individual plots may return figure objects for further customization.
            
        :raises NameError:
            If required data for a requested plot is not available (e.g., requesting
            'emcee_corner' before running run_emcee()).
        :raises AttributeError:
            If the model has not been properly initialized or trained.
            
        Notes
        -----
        Plots are automatically saved to the model's save directory (self.savedir)
        with descriptive filenames. The save directory is created if it doesn't exist.
        
        Training diagnostic plots help assess:
        - Convergence of active learning process
        - Quality of GP hyperparameter optimization  
        - Efficiency of training sample selection
        
        MCMC diagnostic plots help assess:
        - Posterior sampling convergence
        - Chain mixing and autocorrelation
        - Comparison between different sampling methods
        
        Examples
        --------
        Generate all GP training plots:
        
        >>> sm.plot(plots=['gp_all'])
        
        Create MCMC comparison plots:
        
        >>> sm.run_emcee()
        >>> sm.run_dynesty()
        >>> sm.plot(plots=['emcee_corner', 'dynesty_corner', 'mcmc_comparison'])
        
        Generate 2D GP visualization with custom colorbar:
        
        >>> sm.plot(plots=['gp_fit_2D'], cb_rng=[-10, 0], log_scale=True)
        
        Show plots interactively:
        
        >>> sm.plot(plots=['test_mse', 'gp_hyperparameters'], show=True)
        """
        
        # ================================
        # GP training plots
        # ================================

        if "gp_all" in plots:
            gp_plots = ["test_mse", "test_scaled_mse", "test_log_mse", "gp_hyperparam", "gp_timing", "gp_train_scatter"]
            if self.ndim == 2:
                gp_plots.append("gp_fit_2D")
            for pl in gp_plots:
                plots.append(pl)
                
        # GP mean squared error vs iteration
        if "test_mse" in plots:
            
            savename = "gp_mse_vs_iteration.png"
            if hasattr(self, "training_results"):
                print(f"Plotting the gp mean squared error with {self.ntest} test samples...")
                print(f"Saving to {self.savedir}/{savename}")
                
                iarray = np.array(self.training_results["iteration"])
                
                # MSE 
                return vis.plot_error_vs_iteration(iarray,
                                            self.training_results["training_mse"],
                                            self.training_results["test_mse"],
                                            metric="Mean Squared Error",
                                            log=False,
                                            title=f"{self.kernel_name} surrogate",
                                            savedir=self.savedir,
                                            savename=savename,
                                            show=show)
            else:
                raise NameError("Must run active_train before plotting test_mse.")
            
        # GP scaled mean squared error vs iteration
        if "test_scaled_mse" in plots:
            
            savename = "gp_scaled_mse_vs_iteration.png"
            if hasattr(self, "training_results"):
                print(f"Plotting the scaled gp mean squared error with {self.ntest} test samples...")
                print(f"Saving to {self.savedir}/{savename}")

                iarray = np.array(self.training_results["iteration"])
                
                # Scaled MSE
                return vis.plot_error_vs_iteration(iarray,
                                            self.training_results["training_scaled_mse"],
                                            self.training_results["test_scaled_mse"],
                                            metric="Mean Squared Error / Variance",
                                            log=False,
                                            title=f"{self.kernel_name} surrogate",
                                            savedir=self.savedir,
                                            savename=savename,
                                            show=show)
            else:
                raise NameError("Must run active_train before plotting test_scaled_mse.")
            
        # GP mean squared error vs iteration
        if "test_log_mse" in plots:
            
            savename = "gp_mse_vs_iteration_log.png"
            if hasattr(self, "training_results"):
                print(f"Plotting the log gp mean squared error with {self.ntest} test samples...")
                print(f"Saving to {self.savedir}/{savename}")

                iarray = np.array(self.training_results["iteration"])
                
                # Log MSE
                return vis.plot_error_vs_iteration(iarray,
                                            self.training_results["training_mse"],
                                            self.training_results["test_mse"],
                                            metric="Log(Mean Squared Error)",
                                            log=True,
                                            title=f"{self.kernel_name} surrogate",
                                            savedir=self.savedir,
                                            savename=savename,
                                            show=show)
            else:
                raise NameError("Must run active_train before plotting test_log_mse.")

        # GP hyperparameters vs iteration
        if "gp_hyperparameters" in plots:
            if hasattr(self, "training_results"):
                print("Plotting gp hyperparameters...")
                return vis.plot_hyperparam_vs_iteration(self, title=f"{self.kernel_name} surrogate", show=show)
            else:
                raise NameError("Must run active_train before plotting gp_hyperparameters.")

        # GP training time vs iteration
        if "gp_train_time" in plots:
            if hasattr(self, "training_results"):
                print("Plotting gp timing...")
                return vis.plot_train_time_vs_iteration(self, title=f"{self.kernel_name} surrogate", show=show)
            else:
                raise NameError("Must run active_train before plotting gp_timing.")

        # N-D scatterplots and histograms colored by function value
        if "gp_train_corner" in plots:  
            if hasattr(self, "_theta") and hasattr(self, "_y"):
                print("Plotting training sample corner plot...")
                return vis.plot_corner_lnp(self, show=show);
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        # N-D scatterplots and histograms
        if "gp_train_scatter" in plots:  
            if hasattr(self, "_theta") and hasattr(self, "_y"):
                print("Plotting training sample corner plot...")
                return vis.plot_corner_scatter(self, show=show);
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_train_corner.")

        # GP fit (only for 2D functions)
        if "gp_fit_2D" in plots:
            if hasattr(self, "_theta") and hasattr(self, "_y"):
                print("Plotting gp fit 2D...")
                if self.ndim == 2:
                    return vis.plot_gp_fit_2D(self, ngrid=60, title=f"{self.kernel_name} surrogate", show=show, 
                                              vmin=cb_rng[0], vmax=cb_rng[1], log_scale=log_scale)
                else:
                    print("theta must be 2D to use gp_fit_2D!")
            else:
                raise NameError("Must run init_train and/or active_train before plotting gp_fit_2D.")

        # Objective function contour plot
        if "obj_fn_2D" in plots:
            if hasattr(self, "_theta") and hasattr(self, "_y") and hasattr(self, "gp"):
                print("Plotting objective function contours 2D...")
                return vis.plot_utility_2D(self, ngrid=60, show=show, vmin=cb_rng[0], vmax=cb_rng[1], log_scale=log_scale)
            else:
                raise NameError("Must run init_train and init_gp before plotting obj_fn_2D.")
            
        if "true_fn_2D" in plots:
            if self.ndim == 2:
                print("Plotting true function contours 2D...")
                return vis.plot_true_fit_2D(self, ngrid=60, show=show, vmin=cb_rng[0], vmax=cb_rng[1], log_scale=log_scale)
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
                return vis.plot_corner(self, self.emcee_samples, sampler="emcee_", show=show);
            else:
                raise NameError("Must run run_emcee before plotting emcee_corner.")

        # emcee walkers
        if "emcee_walkers" in plots:  
            if hasattr(self, "emcee_samples"):
                print("Plotting emcee walkers...")
                return vis.plot_emcee_walkers(self, show=show)
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
                return vis.plot_corner(self, self.dynesty_samples, sampler="dynesty_", show=show);
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_corner.")

        if "dynesty_corner_kde" in plots:  
            if hasattr(self, "dynesty_samples"):
                print("Plotting dynesty posterior kde...")
                return vis.plot_corner_kde(self, show=show)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_corner.")

        if "dynesty_traceplot" in plots:
            if hasattr(self, "res"):
                print("Plotting dynesty traceplot...")
                return vis.plot_dynesty_traceplot(self, show=show)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_traceplot.")

        if "dynesty_runplot" in plots:
            if hasattr(self, "res"):
                print("Plotting dynesty runplot...")
                return vis.plot_dynesty_runplot(self, show=show)
            else:
                raise NameError("Must run run_dynesty before plotting dynesty_runplot.")

        # ================================
        # MCMC comparison plots
        # ================================

        if "mcmc_comparison" in plots:
            if hasattr(self, "emcee_samples") and hasattr(self, "res"):
                print("Plotting emcee vs dynesty posterior comparison...")
                return vis.plot_emcee_dynesty_comparison(self, show=show)
            else:
                raise NameError("Must run run_emcee and run_dynesty before plotting emcee_comparison.")
    

    def _run_chain_worker(self, chain_id, niter=5, algorithm="bape", gp_opt_freq=1,
                          obj_opt_method="lbfgsb", nopt=10, 
                          use_grad_opt=True, optimizer_kwargs=None, show_progress=True,
                          allow_opt_multiproc=False):
        """
        Worker function to run a single active learning chain.
        This function is designed to be pickled for multiprocessing.
        
        :param chain_id: (*int*)
            Identifier for this chain
        Other parameters same as active_train()
        
        :returns: *tuple or None*
            (new_theta, new_y, training_results) if successful, None if failed
        """
        try:
            # Create a copy of the current model for this chain
            chain = self._create_chain_copy(chain_id=chain_id)
            
            # Store initial state
            initial_theta = chain._theta.copy()
            initial_y = chain._y.copy()
            
            # Run active learning on this chain (explicitly disable multiprocessing)
            chain.active_train(niter=niter, algorithm=algorithm, gp_opt_freq=gp_opt_freq,
                              save_progress=False, obj_opt_method=obj_opt_method, 
                              nopt=nopt, use_grad_opt=use_grad_opt,
                              optimizer_kwargs=optimizer_kwargs, show_progress=show_progress, 
                              allow_opt_multiproc=allow_opt_multiproc)
            
            # Extract only the new samples (excluding initial training data)
            initial_len = len(initial_theta)
            new_theta = chain._theta[initial_len:]
            new_y = chain._y[initial_len:]
            
            return new_theta, new_y, chain.training_results
            
        except Exception as e:
            print(f"Chain {chain_id} failed with error: {e}")
            return None
            
    
    def _create_chain_copy(self, chain_id):
        """Create a copy of the current SurrogateModel for a parallel chain."""
        import copy
        
        # Create a deep copy of the current model
        chain = copy.deepcopy(self)
        
        # Modify savedir to avoid conflicts
        chain.savedir = f"{self.savedir}/chain_{chain_id}"
        if not os.path.exists(chain.savedir):
            os.makedirs(chain.savedir)
        
        # Reset training results for this chain
        chain.training_results = {"iteration" : [], 
                                 "gp_hyperparameters" : [],  
                                 "training_mse" : [],
                                 "test_mse" : [],
                                 "training_scaled_mse" : [],
                                 "test_scaled_mse" : [],
                                 "gp_kl_divergence" : [],
                                 "gp_train_time" : [],
                                 "obj_fn_opt_time" : [],
                                 "gp_hyperparameter_opt_iteration" : [],
                                 "gp_hyperparam_opt_time" : []}
        
        # Ensure chain copies use single-threaded operations
        chain.ncore = 1  # Force single-core for all operations in chain copies
        if hasattr(chain, 'opt_gp_kwargs'):
            chain.opt_gp_kwargs = chain.opt_gp_kwargs.copy()
            chain.opt_gp_kwargs["multi_proc"] = False
            
        # Try to control threading at the library level
        try:
            # Set NumPy to single-threaded if possible
            import numpy as np
            if hasattr(np, 'random') and hasattr(np.random, 'seed'):
                # Some NumPy operations respect thread limits better after reseeding
                np.random.seed(chain_id + 42)  # Different seed per chain
        except:
            pass
            
        try:
            # Control scikit-learn backend
            import sklearn
            if hasattr(sklearn, 'get_config'):
                current_config = sklearn.get_config()
                sklearn.set_config(assume_finite=True, working_memory=128)
        except:
            pass
        
        return chain
    
    
    def _combine_chain_results(self, all_new_theta, all_new_y, chain_results):
        """Combine results from all chains into the main model."""
        
        # Concatenate all new training samples
        if len(all_new_theta) > 0 and any(len(theta) > 0 for theta in all_new_theta):
            combined_new_theta = np.vstack([theta for theta in all_new_theta if len(theta) > 0])
            combined_new_y = np.hstack([y for y in all_new_y if len(y) > 0])
            
            # CRITICAL: Remove near-duplicate points to prevent numerical instability
            # Check for points that are very close to each other (within 1e-6)
            tolerance = 1e-6
            keep_indices = []
            for i, point in enumerate(combined_new_theta):
                is_duplicate = False
                # Check against previously kept points
                for j in keep_indices:
                    if np.allclose(point, combined_new_theta[j], atol=tolerance):
                        is_duplicate = True
                        break
                # Also check against existing training data
                if not is_duplicate:
                    for existing_point in self._theta:
                        if np.allclose(point, existing_point, atol=tolerance):
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    keep_indices.append(i)
            
            if len(keep_indices) < len(combined_new_theta):
                print(f"Warning: Removed {len(combined_new_theta) - len(keep_indices)} near-duplicate points from parallel chains")
                combined_new_theta = combined_new_theta[keep_indices]
                combined_new_y = combined_new_y[keep_indices]
            
            # Add to main model (only if we have points left)
            if len(combined_new_theta) > 0:
                self._theta = np.vstack([self._theta, combined_new_theta])
                self._y = np.hstack([self._y, combined_new_y])
            
                # Update counters
                self.ntrain = len(self._theta)
                self.nactive = self.ntrain - self.ninit_train
                
                # Refit GP with all combined data while preserving hyperparameters  
                # Store current hyperparameters before refitting
                if hasattr(self, 'gp') and self.gp is not None:
                    try:
                        # Refit GP with combined data using current hyperparameters
                        self.gp, _ = self._fit_gp(_theta=self._theta, _y=self._y, hyperparameters=self.gp.get_parameter_vector())
                        
                        # Check for numerical stability by computing a test prediction
                        try:
                            test_pred = self.gp.predict(self._y, self._theta[:5], return_var=False, return_cov=False)
                            if np.any(np.isnan(test_pred)) or np.any(np.isinf(test_pred)):
                                raise ValueError("GP predictions contain NaN or Inf")
                        except:
                            print("Warning: GP numerically unstable after hyperparameter restoration, re-optimizing...")
                            # If unstable, re-optimize hyperparameters
                            self.gp, _ = self._opt_gp(**self.opt_gp_kwargs)
                    except Exception as e:
                        print(f"Warning: Hyperparameter preservation failed ({e}), falling back to standard fitting")
                        # If hyperparameter preservation fails, fall back to standard fitting  
                        self.gp, _ = self._fit_gp(_theta=self._theta, _y=self._y)
                else:
                    # No existing GP, use standard fitting
                    self.gp, _ = self._fit_gp(_theta=self._theta, _y=self._y)
            else:
                print("Warning: No new points added after duplicate removal")
        
        # Combine training results from all chains
        if chain_results:
            self._merge_training_results(chain_results)
        
        # Ensure training results have at least one entry to avoid index errors
        if not self.training_results.get("test_mse"):
            self.training_results["test_mse"] = [np.nan]
        if not self.training_results.get("training_mse"):
            self.training_results["training_mse"] = [np.nan]
        if not self.training_results.get("training_scaled_mse"):
            self.training_results["training_scaled_mse"] = [np.nan]
    
    
    def _merge_training_results(self, chain_results):
        """Merge training results from multiple chains."""
        
        # Get the starting iteration number
        if len(self.training_results["iteration"]) == 0:
            start_iter = 0
        else:
            start_iter = max(self.training_results["iteration"]) + 1
        
        # Merge results from all chains
        for chain_result in chain_results:
            if not chain_result:  # Skip empty results
                continue
                
            for key in self.training_results.keys():
                if key in chain_result:
                    if key == "iteration":
                        # Adjust iteration numbers to be sequential
                        adjusted_iters = [iter_num + start_iter for iter_num in chain_result[key]]
                        self.training_results[key].extend(adjusted_iters)
                        start_iter = max(adjusted_iters) + 1 if adjusted_iters else start_iter
                    else:
                        self.training_results[key].extend(chain_result[key])


    def get_chain_diversity_metrics(self):
        """
        Calculate diversity metrics for the combined training samples.
        Useful for assessing the effectiveness of parallel chains.
        """
        
        if not hasattr(self, '_theta') or len(self._theta) <= self.ninit_train:
            return {}
        
        # Get only the active learning samples (exclude initial training)
        active_theta = self._theta[self.ninit_train:]
        active_y = self._y[self.ninit_train:]
        
        # Calculate diversity metrics
        metrics = {}
        
        # Parameter space coverage
        for i in range(self.ndim):
            param_range = active_theta[:, i].max() - active_theta[:, i].min()
            bound_range = self._bounds[i][1] - self._bounds[i][0]
            metrics[f'param_{i}_coverage'] = param_range / bound_range
        
        # Function value diversity
        if len(active_y) > 1:
            metrics['function_value_std'] = np.std(active_y)
            metrics['function_value_range'] = active_y.max() - active_y.min()
        
        # Average pairwise distance in parameter space
        if len(active_theta) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(active_theta)
            metrics['avg_pairwise_distance'] = np.mean(distances)
            metrics['min_pairwise_distance'] = np.min(distances)
        
        return metrics


    def _fix_pymultinest_output_format(self, outputfiles_basename):
        """
        Fix malformed scientific notation in PyMultiNest output files.
        
        PyMultiNest sometimes writes very small numbers in malformed scientific 
        notation (e.g., '1.23-100' instead of '1.23E-100'), which causes numpy 
        to fail when loading the files. This method fixes such formatting issues.
        
        :param outputfiles_basename: (*str*)
            Base name of PyMultiNest output files to fix
        """
        import re
        import os
        
        # Common PyMultiNest output file extensions that may contain malformed numbers
        suffixes = [
            'post_equal_weights.dat',    # Equal weighted posterior samples
            'phys_live.points',          # Live points
            'post_separate.dat',         # Separate mode samples
            'stats.dat',                 # Statistics file
            'live.points',               # Live points file
            'ev.dat',                    # Evidence file
            'summary.txt',               # Summary file
            '.txt',                      # Generic text output
            'points.dat',                # Points file
            'posterior_samples.dat',     # Alternative posterior samples name
        ]
        
        for suffix in suffixes:
            filepath = f"{outputfiles_basename}{suffix}"
            
            if os.path.exists(filepath):
                try:
                    # Read the file
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Fix malformed scientific notation using regex
                    # Pattern matches various forms of malformed scientific notation:
                    # 1. "1.234567890123456789-123" -> "1.234567890123456789E-123"
                    # 2. "-0.139060048608094152-308" -> "-0.139060048608094152E-308"
                    # 3. "1.23+45" -> "1.23E+45" (if exponent is large)
                    
                    # More comprehensive pattern for malformed scientific notation
                    pattern = r'(-?\d+\.?\d*)([+-]\d+)(?=\s|$|,|\t)'
                    
                    # Function to fix scientific notation
                    def fix_scientific(match):
                        number = match.group(1)
                        exponent = match.group(2)
                        # Only fix if this looks like malformed scientific notation
                        # (exponent magnitude suggests it's not just arithmetic)
                        exp_value = int(exponent)
                        if abs(exp_value) > 10:  # Likely scientific notation
                            return f"{number}E{exponent}"
                        else:
                            return match.group(0)  # Leave unchanged
                    
                    # Apply the fix
                    fixed_content = re.sub(pattern, fix_scientific, content)
                    
                    # Write back only if changes were made
                    if fixed_content != content:
                        with open(filepath, 'w') as f:
                            f.write(fixed_content)
                            
                except Exception as e:
                    if getattr(self, 'verbose', False):
                        print(f"Warning: Could not fix formatting in {filepath}: {e}")

    def _get_pickleable_state(self):
        """
        Get a pickleable representation of the model state for multiprocessing.
        
        :returns: *dict*
            Dictionary containing all necessary state information including GP hyperparameters
        """
        import copy
        
        # Create a simplified state dictionary
        state = {
            'theta': self._theta.copy(),
            'y': self._y.copy(),
            'bounds': copy.deepcopy(self._bounds),
            'ndim': self.ndim,
            'ninit_train': self.ninit_train,
            'ntrain': self.ntrain,
            'function': self.true_log_likelihood,  # Use the correct function attribute
            'verbose': self.verbose,
            'ncore': self.ncore
        }
        
        # Include GP hyperparameters if GP exists and is trained
        if hasattr(self, 'gp') and self.gp is not None:
            try:
                state['gp_hyperparameters'] = self.gp.get_parameter_vector()
                # Also include GP initialization settings for consistency
                state['gp_kernel'] = getattr(self, '_gp_kernel', 'ExpSquaredKernel')
                state['gp_fit_amp'] = getattr(self, '_gp_fit_amp', True)
                state['gp_fit_mean'] = getattr(self, '_gp_fit_mean', True)
                state['gp_fit_white_noise'] = getattr(self, '_gp_fit_white_noise', True)
                state['gp_white_noise'] = getattr(self, '_gp_white_noise', -12)
            except:
                # If we can't get hyperparameters, workers will initialize fresh GP
                state['gp_hyperparameters'] = None
        else:
            state['gp_hyperparameters'] = None
        
        return state


def _run_chain_worker_mp(args):
    """
    Multiprocessing worker function to run a single active learning chain.
    
    This function is designed to work with multiprocessing.Pool and must be
    defined at module level to be pickleable.
    
    :param args: (*tuple*)
        Tuple containing (chain_state, niter, algorithm, gp_opt_freq, 
        obj_opt_method, nopt, use_grad_opt, optimizer_kwargs)
        
    :returns: *tuple or None*
        (new_theta, new_y, training_results) if successful, None if failed
    """
    import os
    import numpy as np
    import copy
    
    try:
        # Unpack arguments
        (chain_state, niter, algorithm, gp_opt_freq, obj_opt_method,
         nopt, use_grad_opt, optimizer_kwargs) = args
        
        chain_id = chain_state['chain_id']
        
        # Reconstruct the model from the pickled state
        from alabi import SurrogateModel
        
        # Create a new model instance with the saved state
        surrogate = SurrogateModel(
            lnlike_fn=chain_state['function'],
            bounds=chain_state['bounds'],
            ncore=1,  # Force single-core for this process
            verbose=False  # Disable verbose to avoid cluttering output
        )
        
        # Restore the training data
        surrogate._theta = chain_state['theta'].copy()
        surrogate._y = chain_state['y'].copy() 
        surrogate.ndim = chain_state['ndim']
        surrogate.ninit_train = chain_state['ninit_train']
        surrogate.ntrain = chain_state['ntrain']
        
        # Set up the save directory for this chain
        surrogate.savedir = chain_state['savedir']
        if not os.path.exists(surrogate.savedir):
            os.makedirs(surrogate.savedir)
        
        # CRITICAL: Set unique random seed for this chain to prevent identical point generation
        import time
        # Use chain_id and current time to ensure unique randomization per chain
        chain_random_seed = int((time.time() * 1000000) % (2**31)) + chain_id * 1000
        np.random.seed(chain_random_seed)
        
        # Initialize GP for this chain using preserved hyperparameters
        use_preserved_hyperparams = (chain_state['gp_hyperparameters'] is not None)
        
        if use_preserved_hyperparams:
            # Initialize GP with same settings as parent process
            surrogate.init_gp(
                kernel=chain_state.get('gp_kernel', 'ExpSquaredKernel'),
                fit_amp=chain_state.get('gp_fit_amp', True),
                fit_mean=chain_state.get('gp_fit_mean', True),
                fit_white_noise=chain_state.get('gp_fit_white_noise', True),
                white_noise=chain_state.get('gp_white_noise', -12)
            )
            # Set the optimized hyperparameters from parent process
            surrogate.gp.set_parameter_vector(chain_state['gp_hyperparameters'])
            # CRITICAL: Compute the GP with the restored hyperparameters
            surrogate.gp.compute(surrogate._theta)
            
            # Add small random perturbation to hyperparameters to prevent identical chains
            # This helps avoid numerical issues while preserving the optimization quality
            current_hyperparams = surrogate.gp.get_parameter_vector()
            # Add 1% random noise to hyperparameters
            perturbation = np.random.normal(0, 0.01 * np.abs(current_hyperparams))
            perturbed_hyperparams = current_hyperparams + perturbation
            surrogate.gp.set_parameter_vector(perturbed_hyperparams)
            surrogate.gp.compute(surrogate._theta)
            
            # IMPROVED: Allow limited GP re-optimization for accuracy
            # Increase gp_opt_freq to reduce re-optimization but don't disable it completely
            # This balances preserved hyperparameters with GP accuracy as data grows
            effective_gp_opt_freq = max(gp_opt_freq * 3, 50)  # At least 3x original freq, min 50
        else:
            # Fallback to fresh initialization if hyperparameters not available
            surrogate.init_gp()
            effective_gp_opt_freq = gp_opt_freq
        
        # Store initial state for comparison
        initial_theta = surrogate._theta.copy()
        initial_y = surrogate._y.copy()
        
        # Run active learning on this chain (force single-core execution)        
        surrogate.active_train(
            niter=niter, 
            algorithm=algorithm, 
            gp_opt_freq=effective_gp_opt_freq,
            save_progress=False,  # Don't save intermediate progress in parallel chains
            obj_opt_method=obj_opt_method, 
            nopt=nopt, 
            use_grad_opt=use_grad_opt,
            optimizer_kwargs=optimizer_kwargs, 
            show_progress=False,  # Don't show progress bars in parallel
            allow_opt_multiproc=False,  # Critical: disable nested multiprocessing
        )
        
        # Extract only the new samples (excluding initial training data)
        initial_len = len(initial_theta)
        new_theta = surrogate._theta[initial_len:]
        new_y = surrogate._y[initial_len:]
        
        return new_theta, new_y, surrogate.training_results
        
    except Exception as e:
        print(f"Multiprocessing chain {chain_state.get('chain_id', '?')} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None