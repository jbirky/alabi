"""
auto_hp_settings
================

This Python script was automatically generated from the Jupyter notebook
auto_hp_settings.ipynb.

You can run this script directly or copy sections into your own code.
"""

# %% [markdown]
# # Automated Hyperparameter Selection
# 
# This tutorial demonstrates how to automatically select optimal Gaussian Process hyperparameters for your surrogate model by testing multiple configurations and choosing the one with the best test set performance.
# 
# ## Why This Matters
# 
# GP surrogate model performance is highly sensitive to:
# - **Kernel choice** (ExpSquared, Matern32, Matern52, etc.)
# - **Data scaling** (no scaling, MinMax, StandardScaler)
# - **Other hyperparameters** (white noise, amplitude, length scales)
# 
# Rather than manually tuning these, we can systematically test combinations and select the best configuration based on test set mean squared error (MSE).

# %% [markdown]
# ## Step 1: Import Required Libraries
# 
# Import `alabi` and its submodules along with standard scientific computing libraries.

# %%
import sys 

sys.path.append("/Users/m1/research/alabi")

import alabi

import alabi.utility as ut

import alabi.metrics as metrics

import alabi.benchmarks as bm

import alabi.visualization as vis

from alabi.core import SurrogateModel



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from functools import partial

from sklearn import preprocessing

from itertools import product

# %% [markdown]
# ## Step 2: Define Problem and Base Configuration
# 
# Set up the benchmark problem (eggbox function) and define a base configuration for GP hyperparameters. These settings will be partially overridden when we test different combinations.
# 
# **Key parameters:**
# - `ninit`: Number of initial training points
# - `niter`: Number of active learning iterations
# - `ncore`: Number of CPU cores for parallel evaluation (use 1 if experiencing multiprocessing issues)
# - `white_noise`: Log-scale white noise parameter (increase if you see positive definiteness errors)

# %%
ninit = 50

niter = 100

basedir = "demo"

kernel = "ExpSquaredKernel"

benchmark = "rosenbrock"

savedir = f"{basedir}/{benchmark}/{kernel}/{ninit}_{niter}"



gp_kwargs = {"kernel": kernel, 

             "fit_amp": True, 

             "fit_mean": True, 

             "fit_white_noise": False, 

             "white_noise": -12,

             "gp_opt_method": "l-bfgs-b",

             "hyperopt_method": "cv",

             "cv_folds": 8,

             "gp_amp_rng": [-1,1],

             "gp_scale_rng": [-2,2],

             "theta_scaler": alabi.no_scaler,

             "y_scaler": alabi.no_scaler}

    

sm = SurrogateModel(lnlike_fn=bm.eggbox["fn"], 

                    bounds=bm.eggbox["bounds"], 

                    savedir=savedir,

                    ncore=8, 

                    verbose=True)



sm.init_samples(ntrain=ninit, ntest=1000, sampler="sobol")

# %% [markdown]
# ## Step 3: Create Hyperparameter Grid
# 
# Generate all combinations of the hyperparameters we want to test:
# - **Kernels**: Different covariance functions capture different smoothness assumptions
# - **Theta scaler**: How to scale input parameters
# - **Y scaler**: How to scale output values
# 
# The `dict_to_combinations` function creates a Cartesian product of all options.

# %%
def dict_to_combinations(options_dict):

    keys = options_dict.keys()

    values = options_dict.values()

    return [dict(zip(keys, combo)) for combo in product(*values)]



def combinations_to_dict(combinations):

    result = {key: [] for key in combinations[0].keys()}

    for combo in combinations:

        for key, value in combo.items():

            result[key].append(value)

    return result



gp_kwarg_options = {"kernel": ["ExpSquaredKernel", "Matern32Kernel", "Matern52Kernel"],

                    "theta_scaler": [ut.no_scaler, preprocessing.MinMaxScaler(), preprocessing.StandardScaler()],

                    "y_scaler": [ut.no_scaler, preprocessing.MinMaxScaler(), preprocessing.StandardScaler()]}



variable_settings = dict_to_combinations(gp_kwarg_options)

print(len(variable_settings), "combinations to test")



# get a list of dictionaries with all combinations of settings, where each dictionary is a copy of the original gp_kwargs with the variable settings updated

setting_combos = []

for settings in variable_settings:

    new_settings = gp_kwargs.copy()

    for key in settings.keys():

        new_settings[key] = settings[key]

    setting_combos.append(new_settings)

# %% [markdown]
# ## Step 4: Test All Hyperparameter Combinations
# 
# Loop through all combinations and fit a GP for each. The `init_gp` method returns the test set MSE, which we use to evaluate performance.
# 
# **Note:** This can take several minutes depending on the number of combinations and problem dimensionality. Use `try/except` to handle configurations that fail to converge.

# %%
for ii in range(len(setting_combos)):

    try:

        test_mse = sm.init_gp(**setting_combos[ii], overwrite=True)

    except:

        test_mse = np.nan

    setting_combos[ii]["test_mse"] = test_mse

# %% [markdown]
# ## Step 5: Analyze Results
# 
# Convert results to a pandas DataFrame and inspect the top-performing configurations. Lower test MSE indicates better generalization performance.

# %%
results = pd.DataFrame(data=combinations_to_dict(setting_combos))

results.sort_values("test_mse").head(10)

# %% [markdown]
# ## Step 6: Extract Best Configuration
# 
# Identify the hyperparameter configuration with the lowest test MSE. This will be used for active learning.

# %%
best_gp_results = results[results["test_mse"] == results["test_mse"].min()]

best_gp_results

# %% [markdown]
# ## Step 7: Run Active Learning
# 
# Use the optimal GP configuration for active learning. The BAPE (Bayesian Active Posterior Estimation) algorithm iteratively selects new training points to improve the surrogate model.
# 
# **Key active learning parameters:**
# - `algorithm`: Acquisition function ("bape", "agp", or "jones")
# - `gp_opt_freq`: How often to reoptimize GP hyperparameters during training
# - `obj_opt_method`: Optimization method for acquisition function
# - `nopt`: Number of optimization restarts for acquisition function

# %%
best_gp_kwargs = best_gp_results[gp_kwargs.keys()].to_dict(orient="records")[0]



al_kwargs = {"algorithm": "bape", 

             "gp_opt_freq": 20, 

             "obj_opt_method": "nelder-mead", 

             "nopt": 6}



sm.init_gp(**best_gp_kwargs, overwrite=True)

sm.active_train(niter=200, **al_kwargs)

# %% [markdown]
# ## Step 8: Visualize Training Progress
# 
# Plot the test set MSE over active learning iterations. A decreasing trend indicates the surrogate model is improving.

# %%
plt.plot(sm.training_results["iteration"], sm.training_results["test_scaled_mse"])

# plt.yscale("log")

plt.show()

# %% [markdown]
# ## Next Steps
# 
# Now that you have an optimized surrogate model, you can:
# 
# 1. **Run posterior sampling** with `sm.run_emcee()` or `sm.run_dynesty()`
# 2. **Visualize the surrogate** using `alabi.visualization` tools
# 3. **Test on new problems** by changing the benchmark function
# 4. **Expand the hyperparameter grid** to include more options like:
#    - Different `white_noise` values
#    - Different `cv_folds` settings
#    - Different data scaling functions for `theta_scaler` or `y_scaler`
