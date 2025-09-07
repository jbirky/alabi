"""
plot_line_fit
=============

This Python script was automatically generated from the Jupyter notebook
plot_line_fit.ipynb.

You can run this script directly or copy sections into your own code.
"""

# %% [markdown]
# # Fit a line to data

# %%
import numpy as np

import matplotlib.pyplot as plt

import alabi

from alabi.core import SurrogateModel

import alabi.visualization as vis

from functools import partial

# %% [markdown]
# ### Generate some synthetic data from the model

# %%
a_true = 2.0

b_true = 1.5



# Define the bounds for the input (a and b)

bounds = [(1.5, 2.5), (1, 2)]



np.random.seed(0)

x = np.sort(10*np.random.rand(50))

yerr = np.random.rand(50)

y = a_true*x + b_true + yerr*np.random.randn(50)



plt.errorbar(x, y, yerr=yerr, fmt="k.")

plt.plot(x, a_true*x + b_true, "r-", label="True line")

plt.xlabel("x", fontsize=20)

plt.ylabel("y", fontsize=20)

plt.show()

# %% [markdown]
# ### Define the likelihood function

# %%
def lnlike(theta, x, y, yerr):

    a, b = theta  

    ypred = a * x + b

    lnl = -0.5 * np.sum(((y - ypred)**2) / (yerr**2))

    return lnl



lnl = partial(lnlike, x=x, y=y, yerr=yerr)

# %%
lnl([a_true, b_true])

# %% [markdown]
# ### Train the surrogate model

# %%
# Create the surrogate model

sm = SurrogateModel(lnlike_fn=lnl, bounds=bounds, savedir="results/linear_fit")



# Initialize samples

sm.init_samples(ntrain=200, sampler="sobol")  



print("Data diagnostics:")

print(f"  theta shape: {sm._theta.shape}")

print(f"  y shape: {sm._y.shape}")

print(f"  y range: [{sm._y.min():.3f}, {sm._y.max():.3f}]")

print(f"  y std: {sm._y.std():.3f}")



gp_kwargs = {"kernel": "ExpSquaredKernel", 

             "fit_amp": True, 

             "fit_mean": True, 

             "fit_white_noise": False, 

             "white_noise": -8,

             "gp_opt_method": "l-bfgs-b",

             "gp_scale_rng": [-2,1],

             "optimizer_kwargs": {"max_iter": 50, "xatol": 1e-3, "fatol": 1e-3, "adaptive": True}}



al_kwargs={"algorithm": "bape", 

           "gp_opt_freq": 20, 

           "obj_opt_method": "nelder-mead", 

           "nopt": 1,

           "optimizer_kwargs": {"max_iter": 50, "xatol": 1e-3, "fatol": 1e-2, "adaptive": True}}



# Initialize the Gaussian Process (GP) surrogate model

sm.init_gp(**gp_kwargs)   



# Train the GP surrogate model with conservative settings

sm.active_train(niter=100, **al_kwargs)

# %%
sm.true_log_likelihood([a_true, b_true]), sm.surrogate_log_likelihood([a_true, b_true])

# %% [markdown]
# ### Plot some diagnostics

# %%
sm.plot(plots=["true_fn_2D"])

# %%
sm.plot(plots=["gp_fit_2D"])

# %%
sm.plot(plots=["obj_fn_2D"])

# %%
nll_true = lambda *args: -sm.true_log_likelihood(*args)

vis.plot_contour_2D(nll_true, sm.bounds, sm.savedir, "true_function.png",  title="True function", ngrid=60, 

                    xlabel=sm.param_names[0], ylabel=sm.param_names[1], log_scale=True)

# %%
nll_pred = lambda *args: -sm.surrogate_log_likelihood(*args)

vis.plot_contour_2D(nll_pred, sm.bounds, sm.savedir, "surrogate_model.png", title="Surrogate Model", ngrid=60, 

                    xlabel=sm.param_names[0], ylabel=sm.param_names[1], log_scale=True)

# %% [markdown]
# Now let's compare how MCMC samples from the true function compare to samples from the `alabi` surrogate model:

# %%
sm.run_dynesty(like_fn=sm.true_log_likelihood)

sm.run_dynesty(like_fn=sm.surrogate_log_likelihood)

# %%
vis.plot_mcmc_comparison(sm.dynesty_samples_true, sm.dynesty_samples_surrogate,

                     param_names=sm.param_names, lw=1.5, colors=["red", "royalblue"],

                     name1="true posterior", name2="surrogate posterior",

                     savedir=sm.savedir, savename="mcmc_comparison.png");

# %%
afit, bfit = np.mean(sm.dynesty_samples_surrogate, axis=0)



plt.errorbar(x, y, yerr=yerr, fmt="k.")

plt.plot(x, a_true*x + b_true, "r-", label="True line")

plt.plot(x, afit*x + bfit, "b-", label="Best fit line")

plt.legend(loc="upper left", fontsize=16)

plt.xlabel("x", fontsize=20)

plt.ylabel("y", fontsize=20)

plt.show()
