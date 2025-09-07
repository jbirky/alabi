"""
plot_function1d
===============

This Python script was automatically generated from the Jupyter notebook
plot_function1d.ipynb.

You can run this script directly or copy sections into your own code.
"""

# %% [markdown]
# # Active Learning in 1D
# 
# In this example, we demonstrate the GP fitting process on a 1D function and visualize the active learning function.

# %%
import numpy as np

import matplotlib.pyplot as plt



from alabi.core import SurrogateModel

import alabi.utility as ut



np.random.seed(7)

# %%
def test1d_fn(theta):

    theta = np.asarray(theta)

    return -np.sin(3*theta) - theta**2 + 0.7*theta



# domain of the function

bounds = [(-2,3)]

# %%
sm = SurrogateModel(lnlike_fn=test1d_fn, 

                    bounds=bounds, 

                    savedir=f"results/test1d")



sm.init_samples(ntrain=5)

sm.init_gp(kernel="ExpSquaredKernel", fit_amp=True, fit_mean=True, white_noise=None)



def bape(xgrid):

    return -np.array([ut.bape_utility(np.array([x]), sm.y, sm.gp, sm.bounds) for x in xgrid])



def agp(xgrid):

    return -np.array([ut.agp_utility(np.array([x]), sm.y, sm.gp, sm.bounds) for x in xgrid])

# %%
def plot_gp_training(sm):



    xgrid = np.arange(sm.bounds[0][0], sm.bounds[0][1]+.1, .01)

    mu, var = sm.gp.predict(sm.y, xgrid, return_cov=False, return_var=True)

    eval_bape = bape(xgrid)

    eval_agp = agp(xgrid)

    opt_bape = xgrid[np.argmax(eval_bape)]

    opt_agp = xgrid[np.argmax(eval_agp)]

    title = f"Ninitial = {sm.ninit_train}, active learning iterations = {len(sm.theta) - sm.ninit_train}"



    fig, axs = plt.subplots(2, 1, figsize=[8,12], sharex=True)

    plt.subplots_adjust(hspace=0)

    axs[0].plot(xgrid, test1d_fn(xgrid), color="k", linestyle="--", label="true function")

    axs[0].scatter(sm.theta, sm.y, color="r", label="GP fit")

    axs[0].plot(xgrid, mu, color="r")

    axs[0].fill_between(xgrid, mu - np.sqrt(var), mu + np.sqrt(var), color="r", alpha=0.2)



    axs[1].plot(xgrid, eval_agp, color="g", label="AGP")

    axs[0].axvline(opt_agp, color="g", linestyle="dotted")

    axs[1].axvline(opt_agp, color="g", linestyle="dotted")

    axs[1].plot(xgrid, eval_bape, color="b", label="BAPE")

    axs[0].axvline(opt_bape, color="b", linestyle="dotted")

    axs[1].axvline(opt_bape, color="b", linestyle="dotted")



    axs[0].set_xlim(sm.bounds[0][0], sm.bounds[0][1])

    axs[0].set_ylabel("GP surrogate model", fontsize=25)

    axs[1].set_ylabel("Active Learning function", fontsize=25)

    axs[1].set_xlabel("x", fontsize=25)

    axs[0].legend(loc="lower left", fontsize=18)

    axs[1].legend(loc="lower left", fontsize=18)

    axs[0].set_title(title, fontsize=25)

    axs[0].minorticks_on()

    axs[1].minorticks_on()

    plt.close()



    return fig

# %%
plot_gp_training(sm)

# %%
sm.active_train(niter=30, algorithm="bape", gp_opt_freq=10)

# %%
sm.run_emcee(nwalkers=20, nsteps=int(5e4), opt_init=False)

# %%
sm.run_dynesty()

# %% [markdown]
# Compare the posterior sampled using `emcee` vs `dynesty`:

# %%
plt.hist(sm.emcee_samples.T[0], bins=50, histtype='step', density=True, label="emcee samples")

plt.hist(sm.dynesty_samples.T[0], bins=50, histtype='step', density=True, label="dynesty samples")

plt.xlabel("$x$", fontsize=25)

plt.legend(loc="upper right", fontsize=18, frameon=False)

plt.minorticks_on()

plt.show()
