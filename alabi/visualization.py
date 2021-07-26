"""
:py:mod:`visualization.py` 
-------------------------------------
"""

import numpy as np
import os
import corner
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
font = {'family' : 'normal',
        'weight' : 'light'}
rc('font', **font)

from dynesty import plotting as dyplot

__all__ = ["plot_error_vs_iteration", 
           "plot_hyperparam_vs_iteration", 
           "plot_train_time_vs_iteration",
           "plot_corner_scatter",
           "plot_train_sample_vs_iteration",
           "plot_gp_fit_1D",
           "plot_gp_fit_2D",
           "plot_true_fit_2D",
           "plot_dynesty_traceplot",
           "plot_dynesty_runplot",
           "plot_mcmc_comparison"]


def plot_error_vs_iteration(sm, log=False, title="GP fit"):

    plt.plot(sm.training_results["iteration"], sm.training_results["training_error"], 
                label='train error')
    plt.plot(sm.training_results["iteration"], sm.training_results["test_error"], 
                label='test error')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.minorticks_on()
    plt.title(title, fontsize=22)
    if log:
        plt.yscale('log')
        plt.savefig(f"{sm.savedir}/gp_error_vs_iteration_log.png")
    else:
        plt.savefig(f"{sm.savedir}/gp_error_vs_iteration.png")
    plt.close()


def plot_hyperparam_vs_iteration(sm, title="GP fit"):

    hp_names = sm.gp.get_parameter_names()
    hp_values = np.array(sm.training_results["gp_hyperparameters"])

    fig = plt.figure(figsize=[8,6])

    # Plot log hyperparameters
    for ii, name in enumerate(hp_names):
        plt.plot(sm.training_results["iteration"], hp_values.T[ii], 
                label=name.replace('_', ' '))
    
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('GP scale hyperparameters', fontsize=18)
    plt.xlim(0, max(sm.training_results["iteration"]))
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.title(title, fontsize=22)
    plt.savefig(f"{sm.savedir}/gp_hyperparameters_vs_iteration.png")
    plt.close()


def plot_train_time_vs_iteration(sm, title="GP fit"):

    plt.plot(sm.training_results["iteration"], sm.training_results["gp_train_time"], 
                label='GP train step')
    plt.plot(sm.training_results["iteration"], sm.training_results["obj_fn_opt_time"], 
                label='Active learning step')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.title(title, fontsize=22)
    plt.minorticks_on()
    plt.savefig(f"{sm.savedir}/gp_train_time_vs_iteration.png")
    plt.close()


def plot_train_sample_vs_iteration(sm):

    yy = -sm.y[sm.ninit_train:] 
    plt.scatter(sm.training_results["iteration"], yy)
    plt.yscale('log')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel(r'$-\ln P$', fontsize=18)
    plt.minorticks_on()
    plt.savefig(f"{sm.savedir}/gp_train_sample_vs_iteration.png")
    plt.close()


def plot_corner_scatter(sm):

    yy = -sm.y 

    fig = corner.corner(sm.theta, c=yy, labels=sm.labels, 
            plot_datapoints=False, plot_density=False, plot_contours=False,
            show_titles=True, title_kwargs={"fontsize": 18}, 
            label_kwargs={"fontsize": 22})

    axes = np.array(fig.axes).reshape((sm.ndim, sm.ndim))
    cb_rng = [yy.min(), yy.max()]

    for yi in range(sm.ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            im = ax.scatter(sm.theta.T[xi], sm.theta.T[yi], c=yy, s=2, cmap='coolwarm', 
                            norm=colors.LogNorm(vmin=min(cb_rng), vmax=max(cb_rng)))

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', anchor=(0,1), 
                        shrink=.7, pad=.1)
    cb.set_label(r'$-\ln P$', fontsize=20, labelpad=-80)
    cb.set_ticks(cb_rng)
    cb.ax.tick_params(labelsize=18)
    fig.savefig(f"{sm.savedir}/gp_training_sample_corner.png")
    plt.close()


def plot_gp_fit_1D(sm, title="GP fit"):

    xarr = np.linspace(sm.bounds[0][0], sm.bounds[0][1], 30)
    mu, var = sm.gp.predict(sm.y, xarr, return_var=True)

    fig, ax = plt.subplots()
    plt.plot(xarr, fn(xarr), color='k', linestyle='--', linewidth=.5)
    ax.fill_between(xarr, mu-var, mu+var, color='r', alpha=.8)
    plt.scatter(sm.theta, sm.y, color='r')
    plt.scatter(sm.theta_test, sm.y_test, color='g')
    plt.xlim(sm.bounds[0])
    plt.title(title, fontsize=22)
    plt.savefig(f"{sm.savedir}/gp_fit_1D.png")
    plt.close()


def plot_gp_fit_2D(sm, ngrid=60, title="GP fit"):

    xarr = np.linspace(sm.bounds[0][0], sm.bounds[0][1], ngrid)
    yarr = np.linspace(sm.bounds[1][0], sm.bounds[1][1], ngrid)

    X, Y = np.meshgrid(xarr, yarr)
    Z = np.zeros((ngrid, ngrid))
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            tt = np.array([X[i][j], Y[i][j]]).reshape(1,-1)
            Z[i][j] = sm.evaluate(tt)
        
    im = plt.contourf(X, Y, Z, 20, cmap='Blues_r')
    plt.colorbar(im)
    plt.scatter(sm.theta.T[0], sm.theta.T[1], color='r', s=5)
    plt.title(title, fontsize=22)
    plt.savefig(f"{sm.savedir}/gp_fit_2D.png")
    plt.close()


def plot_true_fit_2D(fn, bounds, savedir, ngrid=60):

    xarr = np.linspace(bounds[0][0], bounds[0][1], ngrid)
    yarr = np.linspace(bounds[1][0], bounds[1][1], ngrid)

    X, Y = np.meshgrid(xarr, yarr)
    Z = np.zeros((ngrid, ngrid))
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            tt = np.array([X[i][j], Y[i][j]])
            Z[i][j] = fn(tt)
        
    im = plt.contourf(X, Y, Z, 20, cmap='Blues_r')
    plt.colorbar(im)
    plt.title("True function", fontsize=22)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(f"{savedir}/true_function_2D.png")
    plt.close()


def plot_corner(sm, samples, sampler=""):

    fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        scale_hist=True, plot_contours=True, labels=sm.labels,
                        title_kwargs={"fontsize": 15}, label_kwargs={"fontsize": 15})
    fig.savefig(f"{sm.savedir}/{sampler}posterior.png", bbox_inches="tight")


def plot_corner_kde(sm):

    fig, axes = dyplot.cornerplot(sm.res, quantiles=[0.16, 0.5, 0.84], span=sm.bounds,
                               title_kwargs={"fontsize": 15}, label_kwargs={"fontsize": 15})
    fig.savefig(f"{sm.savedir}/dynesty_posterior_kde.png", bbox_inches="tight")


def plot_emcee_walkers(sm):

    fig, axes = plt.subplots(sm.ndim, figsize=(12, 3*sm.ndim), sharex=True)
    samples = sm.sampler.get_chain()
    for i in range(sm.ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(sm.labels[i], fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number", fontsize=20)

    fig.savefig(f"{sm.savedir}/emcee_walkers.png", bbox_inches="tight")


def plot_dynesty_traceplot(sm):

    fig, axes = dyplot.traceplot(sm.res, trace_cmap='plasma',
                                 quantiles=None, show_titles=True,
                                 label_kwargs={"fontsize": 22})
    fig.savefig(f"{sm.savedir}/dynesty_traceplot.png")


def plot_dynesty_runplot(sm):

    fig, axes = dyplot.runplot(sm.res, label_kwargs={"fontsize": 22})
    fig.savefig(f"{sm.savedir}/dynesty_runplot.png")


def plot_mcmc_comparison(sm):

    lw = 1.5
    colors = ["orange", "royalblue"]

    fig = corner.corner(sm.emcee_samples,  labels=sm.labels, range=sm.bounds,
                    show_titles=True, verbose=False, max_n_ticks=4,
                    plot_contours=True, plot_datapoints=True, plot_density=True,
                    color=colors[0], no_fill_contours=False, title_kwargs={"fontsize": 16},
                    label_kwargs={"fontsize": 22}, hist_kwargs={"linewidth":2.0, "density":True})

    fig = corner.corner(sm.dynesty_samples, labels=sm.labels, range=sm.bounds, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, verbose=False, max_n_ticks=4, title_fmt='.3f',
                        plot_contours=True, plot_datapoints=True, plot_density=True,
                        color=colors[1], no_fill_contours=False, title_kwargs={"fontsize": 16},
                        label_kwargs={"fontsize": 22}, hist_kwargs={"linewidth":2.0, "density":True},
                        fig=fig)

    fig.axes[1].text(2.2, 0.725, r"--- emcee posterior", fontsize=26, color=colors[0], ha='left')
    fig.axes[1].text(2.2, 0.55, r"--- dynesty posterior", fontsize=26, color=colors[1], ha='left')
    
    fig.savefig(f"{sm.savedir}/mcmc_comparison.png")