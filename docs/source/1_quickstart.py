"""
1_quickstart
============

This Python script was automatically generated from the Jupyter notebook
1_quickstart.ipynb.

You can run this script directly or copy sections into your own code.
"""

# %% [markdown]
# # Quickstart

# %% [markdown]
# ### Step 1
# 
# Import python modules:

# %%
import numpy as np

import matplotlib.pyplot as plt



from alabi.core import SurrogateModel



from matplotlib import rcParams

# rcParams['font.family'] = 'serif'

# rcParams['text.usetex'] = True



random_state = 7

np.random.seed(random_state)

# %% [markdown]
# ### Step 2
# 
# Define the test function and the bounds for the input space. For example:

# %%
def test1d_fn(x):

    return np.sin(5 * x) * (1 - np.tanh(x**2))



bounds = [(-1, 1)]

# %% [markdown]
# ### Step 3
# 
# Initialize the surrogate model, specifying the function to train on, the bounds of the input space, and directory where the results will be saved:

# %%
sm = SurrogateModel(lnlike_fn=test1d_fn, bounds=bounds, savedir=f"results/test1d", random_state=random_state)

# %% [markdown]
# ### Step 4
# 
# Initialize the gaussian process surrogate model by specifying a kernel. In this example we'll use a squared exponential kernel:
# 
# $ k(x, x') = \sigma_f^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right) $
# 
# where $ k(x, x') $ is the kernel function, $ \sigma_f^2 $ is the amplitude hyperparameter, $ \ell $ is the length scale hyperparameter, and $ x $ and $ x' $ are input points.

# %%
sm.init_samples(ntrain=10)

sm.init_gp(kernel="ExpSquaredKernel", fit_amp=True, fit_mean=True, hyperopt_method="ml", overwrite=True)

# %% [markdown]
# ### Step 5
# 
# Improve the surrogate model fit by iteratively selecting new training points using active learning:

# %%
sm.active_train(niter=30, algorithm="bape", gp_opt_freq=10)

# %% [markdown]
# ### Step 6
# 
# Run Markov Chain Monte Carlo (MCMC) sampler using either the `emcee` package:

# %%
sm.run_emcee(nwalkers=4, nsteps=int(5e4))

# %%
sm.run_dynesty()

# %%
xgrid = np.linspace(bounds[0][0], bounds[0][1], 100)

ygrid = np.exp(test1d_fn(xgrid))



plt.hist(sm.emcee_samples.T[0], bins=50, histtype='step', density=True, label="emcee samples")

plt.hist(sm.dynesty_samples.T[0], bins=50, histtype='step', density=True, label="dynesty samples")

plt.plot(xgrid, ygrid/max(ygrid), label="true function", color="k", linestyle="--")

plt.xlabel("$x$", fontsize=25)

plt.xlim(*bounds)

plt.legend(loc="upper left", fontsize=18, frameon=False)

plt.minorticks_on()

plt.show()
