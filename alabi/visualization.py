"""
:py:mod:`visualization.py` 
-------------------------------------
"""

import alabi.utility as ut

import numpy as np
import os
import corner
import warnings
from functools import partial
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
           "plot_corner_lnp",
           "plot_corner_scatter",
           "plot_train_sample_vs_iteration",
           "plot_gp_fit_1D",
           "plot_gp_fit_2D",
           "plot_contour_2D",
           "plot_true_fit_2D",
           "plot_utility_2D",
           "plot_dynesty_traceplot",
           "plot_dynesty_runplot",
           "plot_mcmc_comparison",
           "plot_2D_panel4"]


def plot_error_vs_iteration(iteration, train_error, test_error=None, log=False, 
                            metric="Error", title="GP fit", show=False, 
                            savedir=".", savename=None):

    fig = plt.figure()
    plt.plot(iteration, train_error,  label='train error')
    if test_error is not None:
        plt.plot(iteration, test_error,  label='test error')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.xlim(0, max(iteration))
    plt.minorticks_on()
    plt.title(title, fontsize=22)
    plt.tight_layout()
    
    if savename is None:
        savename = "gp_error_vs_iteration_log.png" if log else "gp_error_vs_iteration.png"
    
    if log:
        plt.yscale('log')

    plt.savefig(f"{savedir}/{savename}")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_hyperparam_vs_iteration(sm, title="GP fit", show=False):

    hp_names = sm.gp.get_parameter_names()
    hp_values = np.array(sm.training_results["gp_hyperparameters"])

    fig, ax1 = plt.subplots(1,1)

    # Plot log hyperparameters
    if sm.fit_mean == True:
        for ii in range(1, len(hp_names)):
            ax1.plot(sm.training_results["iteration"], hp_values.T[ii], 
                    label=hp_names[ii].replace('_', ' '))
        ax1.tick_params(axis='y')
        ax1.fill_between(sm.training_results["iteration"], -sm.gp_scale_rng,
                         sm.gp_scale_rng, color="C2", alpha=0.1, label="GP scale range")

        # plot mean on separate axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('mean hyperparameter', color='grey', fontsize=18)  
        ax2.plot(sm.training_results["iteration"], hp_values.T[0], color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.minorticks_on()
    else:
        for ii, name in enumerate(hp_names):
            ax1.plot(sm.training_results["iteration"], hp_values.T[ii], 
                     label=hp_names[ii].replace('_', ' '))

    ax1.set_xlabel('iteration', fontsize=18)
    ax1.set_ylabel('GP scale hyperparameters', fontsize=18)
    ax1.set_xlim(1, max(sm.training_results["iteration"]))
    # ax1.set_ylim(-1.2*sm.gp_scale_rng, 1.2*sm.gp_scale_rng)
    ax1.minorticks_on()
    ax1.legend(loc='best')
    ax1.set_title(title, fontsize=22)
    plt.tight_layout()
    plt.savefig(f"{sm.savedir}/gp_hyperparameters_vs_iteration.png")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_train_time_vs_iteration(sm, title="GP fit", show=False):

    fig = plt.figure()
    plt.plot(sm.training_results["iteration"], sm.training_results["gp_train_time"], 
                label='GP train step')
    plt.plot(sm.training_results["iteration"], sm.training_results["obj_fn_opt_time"], 
                label='Active learning step')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.title(title, fontsize=22)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(f"{sm.savedir}/gp_train_time_vs_iteration.png")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_train_sample_vs_iteration(sm, show=False):

    fig = plt.figure()

    yy = sm.y()[sm.ninit_train:]
    plt.scatter(sm.training_results["iteration"], yy)
    plt.yscale('log')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel(r'$-\ln P$', fontsize=18)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(f"{sm.savedir}/gp_train_sample_vs_iteration.png")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_corner_lnp(sm, show=False):

    theta = sm.theta()
    yy = sm.y()

    warnings.simplefilter("ignore")
    fig = corner.corner(theta, c=yy, labels=sm.labels, 
            plot_datapoints=False, plot_density=False, plot_contours=False,
            show_titles=True, title_kwargs={"fontsize": 18}, 
            label_kwargs={"fontsize": 22}, data_kwargs={'alpha':1.0})

    axes = np.array(fig.axes).reshape((sm.ndim, sm.ndim))
    cb_rng = [yy.min(), yy.max()]

    for yi in range(sm.ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            im = ax.scatter(theta[xi], theta[yi], c=yy, s=2, cmap='coolwarm', 
                            norm=colors.LogNorm(vmin=min(cb_rng), vmax=max(cb_rng)),
                            alpha=1.0)

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', anchor=(0,1), 
                        shrink=.7, pad=.1)
    cb.set_label(r'$-\ln P$', fontsize=20, labelpad=-80)
    cb.set_ticks(cb_rng)
    cb.ax.tick_params(labelsize=18)
    fig.savefig(f"{sm.savedir}/gp_training_sample_corner.png")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_corner_scatter(sm, show=False):

    theta = sm.theta()
    yy = -sm.y()

    warnings.simplefilter("ignore")
    fig = corner.corner(theta[0:sm.ninit_train], labels=sm.labels, 
            plot_datapoints=True, plot_density=False, plot_contours=False,
            show_titles=True, title_kwargs={"fontsize": 18}, color='b',
            label_kwargs={"fontsize": 22}, data_kwargs={'alpha':1.0})

    if sm.nactive > sm.ndim:
        fig = corner.corner(theta[sm.ninit_train:], labels=sm.labels, 
                plot_datapoints=True, plot_density=False, plot_contours=False,
                show_titles=True, title_kwargs={"fontsize": 18}, color='r',
                label_kwargs={"fontsize": 22}, data_kwargs={'alpha':1.0},
                fig=fig)

    fig.savefig(f"{sm.savedir}/gp_training_sample_scatter.png")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_gp_fit_1D(sm, title="GP fit", show=False):

    theta = sm.theta()
    yy = sm.y()
    theta_test = sm.theta_scaler.inverse_transform(sm.theta_test)
    yy_test = sm.y_scaler.inverse_transform(sm.y_test.reshape(-1, 1)).flatten()
    
    xarr = np.linspace(sm.bounds[0][0], sm.bounds[0][1], 30)
    mu, var = sm.gp.predict(sm.y, xarr, return_var=True)

    fig, ax = plt.subplots()
    plt.plot(xarr, fn(xarr), color='k', linestyle='--', linewidth=.5)
    ax.fill_between(xarr, mu-var, mu+var, color='r', alpha=.8)
    plt.scatter(theta, yy, color='r')
    plt.scatter(theta_test, yy_test, color='g')
    plt.xlim(sm.bounds[0])
    plt.title(title, fontsize=22)
    plt.tight_layout()
    plt.savefig(f"{sm.savedir}/gp_fit_1D.png")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_gp_fit_2D(sm, ngrid=60, title="GP fit", show=False):

    theta = sm.theta() 
    theta0 = sm.theta_scaler.inverse_transform(sm._theta0)

    xarr = np.linspace(sm.bounds[0][0], sm.bounds[0][1], ngrid)
    yarr = np.linspace(sm.bounds[1][0], sm.bounds[1][1], ngrid)

    X, Y = np.meshgrid(xarr, yarr)
    Z = np.zeros((ngrid, ngrid))
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            tt = np.array([X[i][j], Y[i][j]]).reshape(1,-1)
            Z[i][j] = sm.evaluate(tt)
        
    fig = plt.figure()
    im = plt.contourf(X, Y, Z, 20, cmap='Blues_r')
    plt.colorbar(im)
    plt.scatter(theta.T[0], theta.T[1], color='red', edgecolor='none', 
                s=12, label=f'{sm.algorithm} training')
    plt.scatter(theta0.T[0], theta0.T[1], color='#1cc202', edgecolor='none', 
                s=12, label='initial training')
    plt.title(title, fontsize=22)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{sm.savedir}/gp_fit_2D.png")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_contour_2D(fn, bounds, savedir, save_name, title, ngrid=60, cmap='Blues_r', show=False):

    xarr = np.linspace(bounds[0][0], bounds[0][1], ngrid)
    yarr = np.linspace(bounds[1][0], bounds[1][1], ngrid)

    X, Y = np.meshgrid(xarr, yarr)
    Z = np.zeros((ngrid, ngrid))
    
    fig = plt.figure()
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            tt = np.array([X[i][j], Y[i][j]])
            Z[i][j] = fn(tt)
        
    im = plt.contourf(X, Y, Z, 20, cmap=cmap)
    plt.colorbar(im)
    plt.title(title, fontsize=22)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.tight_layout()
    plt.savefig(f"{savedir}/{save_name}")
    if show:
        plt.show()
    plt.close()

    return fig


def plot_true_fit_2D(sm, ngrid=60, show=False):

    fig = plot_contour_2D(sm.fn, sm.bounds, sm.savedir, save_name="true_function_2D.png", 
                    title="True function", ngrid=ngrid)
    if show:
        plt.show()

    return fig


def plot_utility_2D(sm, ngrid=60, show=False):

    ut_fn = ut.assign_utility(sm.algorithm)
    fn = partial(ut_fn, y=sm._y, gp=sm.gp, bounds=sm.bounds)

    fig = plot_contour_2D(fn, sm.bounds, sm.savedir, save_name="objective_function.png", 
                    title=f"{sm.algorithm.upper()} function", ngrid=ngrid, cmap='Greens_r')

    if show:
        plt.show()

    return fig


def plot_corner(sm, samples, sampler="", show=False):

    warnings.simplefilter("ignore")
    fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        scale_hist=True, plot_contours=True, labels=sm.labels,
                        title_kwargs={"fontsize": 20}, label_kwargs={"fontsize": 20})
    fig.savefig(f"{sm.savedir}/{sampler}_posterior_{sm.like_fn_name}.png", bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_corner_kde(sm, show=False):

    fig, axes = dyplot.cornerplot(sm.res, quantiles=[0.16, 0.5, 0.84], span=sm.bounds,
                               title_kwargs={"fontsize": 15}, label_kwargs={"fontsize": 15})
    fig.savefig(f"{sm.savedir}/dynesty_posterior_kde_{sm.like_fn_name}.png", bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_emcee_walkers(sm, show=False):

    fig, axes = plt.subplots(sm.ndim, figsize=(12, 3*sm.ndim), sharex=True)
    samples = sm.emcee_samples_full
    for i in range(sm.ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(sm.labels[i], fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number", fontsize=20)

    fig.savefig(f"{sm.savedir}/emcee_walkers_{sm.like_fn_name}.png", bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_dynesty_traceplot(sm, show=False):

    fig, axes = dyplot.traceplot(sm.res, trace_cmap='plasma',
                                 quantiles=None, show_titles=True,
                                 label_kwargs={"fontsize": 22})
    fig.savefig(f"{sm.savedir}/dynesty_traceplot_{sm.like_fn_name}.png")

    if show:
        plt.show()

    return fig


def plot_dynesty_runplot(sm, show=False):

    fig, axes = dyplot.runplot(sm.res, label_kwargs={"fontsize": 22})
    fig.savefig(f"{sm.savedir}/dynesty_runplot_{sm.like_fn_name}.png")

    if show:
        plt.show()

    return fig


def plot_mcmc_comparison(sm, show=False):

    lw = 1.5
    colors = ["orange", "royalblue"]

    warnings.simplefilter("ignore")
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

    fig.savefig(f"{sm.savedir}/mcmc_comparison_{sm.like_fn_name}.png")

    if show:
        plt.show()

    return fig


def plot_2D_panel4(savedir, save_name=None):

    from PIL import Image
    img_01 = Image.open(f"{savedir}/gp_fit_2D.png")
    img_02 = Image.open(f"{savedir}/objective_function.png")
    img_03 = Image.open(f"{savedir}/gp_error_vs_iteration.png")
    img_04 = Image.open(f"{savedir}/gp_hyperparameters_vs_iteration.png")

    new_im = Image.new("RGB", (2*img_01.size[0], 2*img_01.size[1]), (250,250,250))

    new_im.paste(img_01, (0, 0))
    new_im.paste(img_02, (img_01.size[0], 0))
    new_im.paste(img_03, (0, img_01.size[1]))
    new_im.paste(img_04, (img_01.size[0], img_01.size[1]))

    if save_name is not None:
        new_im.save(f"{savedir}/{save_name}")

    return new_im