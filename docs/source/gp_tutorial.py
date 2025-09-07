"""
gp_tutorial
===========

This Python script was automatically generated from the Jupyter notebook
gp_tutorial.ipynb.

You can run this script directly or copy sections into your own code.
"""

# %% [markdown]
# ## GP Training
# 
# This notebook demonstrates how to train a Gaussian process surrogate model using `alabi` and visualize the results.

# %%
%matplotlib inline

import numpy as np

from alabi import SurrogateModel

# %% [markdown]
# ### Define a Test Problem: Rosenbrock Function
# 
# We'll use the 2D Rosenbrock function as our test case. This is a common benchmark function for optimization and sampling algorithms.

# %%
from scipy.optimize import rosen



def rosenbrock_fn(x):

    return -rosen(x)/100.0



bounds = [(-5,5), (-5,5)]

param_names = ['x1', 'x2']


# %% [markdown]
# ### Create and Train Surrogate Model
# 
# First, we'll create a Gaussian Process surrogate model and train it using active learning.

# %%
# Initialize the surrogate model

sm = SurrogateModel(

    lnlike_fn=rosenbrock_fn,

    bounds=bounds,

    savedir="results/rosenbrock_2d",

    cache=True,

    verbose=True,

    ncore=4

)



# Compute initial training samples (parallelized if ncore > 1)

sm.init_samples(ntrain=100, ntest=1000, sampler="sobol")

# %%
# Initialize the Gaussian Process with specified hyperparameters

gp_kwargs = {"kernel": "ExpSquaredKernel", 

             "fit_amp": True, 

             "fit_mean": True, 

             "fit_white_noise": False, 

             "white_noise": -12,

             "gp_opt_method": "l-bfgs-b",

             "gp_scale_rng": [-2,2],

             "optimizer_kwargs": {"max_iter": 50}}



sm.init_gp(**gp_kwargs)

# %%
al_kwargs={"algorithm": "bape", 

           "gp_opt_freq": 20, 

           "obj_opt_method": "nelder-mead", 

           "use_grad_opt": True,

           "nopt": 1,

           "optimizer_kwargs": {"max_iter": 50, "xatol": 1e-3, "fatol": 1e-2, "adaptive": True}}



sm.active_train(niter=100, **al_kwargs)

# %% [markdown]
# Now you can use the trained GP surrogate model by calling the function `sm.surrogate_log_likelihood(theta)`

# %%
theta_test = np.array([0.0, 0.0])



ytrue = sm.true_log_likelihood(theta_test)

ysurrogate = sm.surrogate_log_likelihood(theta_test)



print(f"True log-likelihood at {theta_test}: {ytrue}")

print(f"Surrogate log-likelihood at {theta_test}: {ysurrogate}")

# %% [markdown]
# ### Visualize the Surrogate Model
# 
# Let's plot the surrogate model to see how well it captures the true function.

# %%
sm.plot(plots=["true_fn_2D"])

# %%
sm.plot(plots=["gp_fit_2D"])

# %% [markdown]
# We can also check how the active learning function looks at this iteration. If the model is closed to converged, we expect the high probability regions to be close to 0.

# %%
sm.plot(plots=["obj_fn_2D"])

# %% [markdown]
# `alabi` also tracks various performance metrics as a function of active learning iteration. These results can be accessed from the dictionary:

# %%
sm.training_results.keys()

# %% [markdown]
# Here are some examples of quick plots you can make of these results:

# %%
sm.plot(plots=["gp_hyperparameters"])

# %%
sm.plot(plots=["test_mse"])

# %%
sm.plot(plots=["test_scaled_mse"])
