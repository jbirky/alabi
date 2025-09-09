"""
plot_kl_vs_iteration_2d
=======================

This Python script was automatically generated from the Jupyter notebook
plot_kl_vs_iteration_2d.ipynb.

You can run this script directly or copy sections into your own code.
"""

# %% [markdown]
# # KL divergence: non-Gaussian
# 
# In this example, we show how to test surrogate model covergence using the KL divergence metric for any general distribution.

# %%
import alabi

import alabi.utility as ut

import alabi.metrics as metrics

import alabi.benchmarks as bm

import alabi.visualization as vis

from alabi.core import SurrogateModel

from alabi.metrics import kl_divergence_kde, compute_kl_single_trial_joblib, compute_kl_full_parallel



import numpy as np

import matplotlib.pyplot as plt

from functools import partial

import os

from joblib import Parallel, delayed

# %% [markdown]
# ## Run functions
# 
# First we'll define a run function for `alabi` that runs MCMC every `niter_per_batch` iterations

# %%
def run_alabi(lnlike_fn, bounds, kernel="ExpSquaredKernel", savedir="results/",

              ntrain=50, nbatch=50, niter_per_batch=10):

    

    sm = SurrogateModel(lnlike_fn=lnlike_fn, bounds=bounds, ncore=ncore, savedir=savedir)

    sm.init_samples(ntrain=ntrain)

    sm.init_gp(kernel=kernel, fit_amp=True, fit_mean=True, white_noise=-12)

    

    # zeroth iteration

    sm.run_dynesty(like_fn=sm.surrogate_log_likelihood)

            

    for _ in range(nbatch):

        sm.active_train(niter=niter_per_batch, algorithm="bape", gp_opt_freq=10)

        try:

            sm.run_dynesty(like_fn=sm.surrogate_log_likelihood)

        except Exception as e:

            print(f"Error occurred while running dynesty: {e}")

            

    return sm

# %% [markdown]
# Next we'll define a wrapper around a function we want to test. In this example we'll use the eggbox function from Birky et al. 2025

# %%
def run_single_trial(ntrain, kernel, trial, alabi_kwargs):

    """

    Run a single trial for all benchmark functions

    """

    base_dir = f"results_2d_init{ntrain}_mcmc/"

    

    sm = run_alabi(bm.eggbox["fn"], bounds=bm.eggbox["bounds"], kernel=kernel, ntrain=ntrain,

                savedir=f"{base_dir}/eggbox/{kernel}/{trial}", **alabi_kwargs)

    sm.savedir = f"{base_dir}/eggbox/{kernel}"

    sm.run_dynesty(like_fn=sm.true_log_likelihood)

# %% [markdown]
# ## Configure a set of tests

# %%
alabi_kwargs = {

    "nbatch": 25,           # total number of times we will compute KL divergence for a run

    "niter_per_batch": 10,  # how frequently (in number of iterations) to compute KL divergence

}



ntrials = 10        # how many trials per setting to average over

ncore = 8           # number of CPU cores to use 



# define which kernels we want to test

kernels = ["ExpSquaredKernel", "Matern52Kernel"]

# in this example we'll test two different kernels, but you can add more to the list



# define how many initial training samples to test

ntrain_list = [50]

# in this example we'll just test one setting, but you can add more to the list, e.g., ntrain_list = [20, 50, 100, 200]



# Create all task combinations

tasks = []

for ntrain in ntrain_list:

    for kernel in kernels:

        for trial in range(ntrials):

            tasks.append((ntrain, kernel, trial, alabi_kwargs))



print("Total number of configurations to run:", len(tasks))

print(f"Total number of iterations per run: {alabi_kwargs['nbatch'] * alabi_kwargs['niter_per_batch']}")

print(f"Total number of runs to compute average KL divergence: {ntrials}")

print(f"Total number of CPU cores available: {ncore}")

# %% [markdown]
# These are all of the different configurations that we will try. For each configuration we will run `ntrials` and compute the average kl divergence as a function of iteration, sampled every `niter_per_batch` iterations, `nbatch` times

# %%
tasks

# %% [markdown]
# ## Execute Trials

# %%
results = Parallel(n_jobs=ncore)(

    delayed(run_single_trial)(ntrain, kernel, trial, alabi_kwargs)

    for ntrain, kernel, trial, alabi_kwargs in tasks

)

# %% [markdown]
# ## Read results into a dictionary

# %%
# ======================================================

# Parallel processing for all configurations with joblib 

# ======================================================



# base_dirs = [f"results_2d_init{ii}_mcmc" for ii in np.arange(10,60,10)]

base_dirs = ["results_2d_init200_mcmc"]

# examples = ["gaussian_2d", "rosenbrock", "gaussian_shells", "eggbox"]

examples = ["eggbox"]

kernels = ["ExpSquaredKernel", "Matern32Kernel", "Matern52Kernel"]

trials = np.arange(0, 30)

iterations = np.arange(10, 250, 10)

kl_results = {}



for base_dir in base_dirs:

    for example in examples:

        for kernel in kernels:

            

            # Check if true samples exist, create if needed

            if not os.path.exists(f"{base_dir}/{example}/{kernel}/dynesty_samples_final_true.npz"):

                sm = alabi.load_model_cache(f"{base_dir}/{example}/{kernel}/0")

                sm.run_dynesty(like_fn=sm.true_log_likelihood)

            

            # Fully parallel processing

            kl_results[f"{base_dir}/{example}/{kernel}"] = compute_kl_full_parallel(

                base_dir, example, kernel, trials, iterations, n_jobs=ncore

            )

# %% [markdown]
# ## Plot results

# %%
colors = ["royalblue", "firebrick", "limegreen"]



for base_dir in base_dirs:

    

    ninit = base_dir.split("_mcmc")[0].split("init")[1]

    for example in examples:

        

        plt.figure(figsize=(8, 6))



        for ii, kernel in enumerate(kernels):

            

            key = f"{base_dir}/{example}/{kernel}"

            if key not in kl_results:

                print(f"Skipping {key} as no results found")

                continue

            

            kl_avg, kl_std, kl_25, kl_50, kl_75 = kl_results[key].T

            

            # plt.plot(iterations, kl_avg, label=f"{kernel}", color=colors[ii])

            # plt.fill_between(iterations, kl_avg - kl_std, kl_avg + kl_std, alpha=0.1, color=colors[ii])

            plt.plot(iterations, kl_50, label=f"{kernel}", color=colors[ii])

            plt.fill_between(iterations, kl_25, kl_75, alpha=0.05, color=colors[ii])

            plt.axhline(0, color="black", linestyle="--", linewidth=1)

            

        plt.legend(loc="upper right", fontsize=18)

        plt.xlabel("Iterations", fontsize=20)

        plt.ylabel("KL Divergence", fontsize=20)

        plt.xlim(iterations[0], iterations[-1])

        plt.yscale("log")

        plt.ylim(1e-3, 1e1)

        

        # Set custom y-axis tick labels

        plt.yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1], ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$'])

        

        plt.minorticks_on()

        plt.title(f"{example} (Ninit={ninit})", fontsize=22)

        # plt.savefig(f"plots/scaling_2d/kl_results_{base_dir}_{example}.png", bbox_inches="tight", dpi=300)

        plt.show()

        plt.close()

# %%
plt.figure(figsize=(10, 6))

for key in kl_results.keys():

    plt.plot(iterations, kl_results[key], label=key.split("/")[-1])

plt.legend(loc='upper right', fontsize=18)

plt.xlim(iterations[0], iterations[-1])

plt.minorticks_on()

plt.show()
