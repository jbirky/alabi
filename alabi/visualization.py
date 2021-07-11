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

__all__ = ["plot_error_vs_iteration", 
           "plot_hyperparam_vs_iteration", 
           "plot_train_time_vs_iteration",
           "plot_corner_scatter",
           "plot_gp_fit_1D",
           "plot_gp_fit_2D",
           "plot_true_fit_2D"]


def plot_error_vs_iteration(training_results, savedir, log=False, title="GP fit"):

    plt.plot(training_results["iteration"], training_results["training_error"], 
                label='train error')
    plt.plot(training_results["iteration"], training_results["test_error"], 
                label='test error')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.minorticks_on()
    plt.title(title, fontsize=22)
    if log:
        plt.yscale('log')
        plt.savefig(f"{savedir}/gp_error_vs_iteration_log.png")
    else:
        plt.savefig(f"{savedir}/gp_error_vs_iteration.png")
    plt.close()


def plot_hyperparam_vs_iteration(training_results, hp_names, 
                                 hp_values, savedir, title="GP fit"):

    fig, axs = plt.subplots(2,1, figsize=[8,10], sharex=True)

    # Plot mean value on separate panel
    axs[0].plot(training_results["iteration"], hp_values.T[0], 
                label=hp_names[0].replace('_', ' '), color='k')
    axs[0].set_ylabel('GP mean hyperparameter', fontsize=18)

    # Plot log hyperparameters
    for ii, name in enumerate(hp_names[1:]):
        axs[1].plot(training_results["iteration"], hp_values.T[ii+1], 
                label=name.replace('_', ' '))
    
    axs[1].set_xlabel('iteration', fontsize=18)
    axs[1].set_ylabel('GP scale hyperparameters', fontsize=18)
    axs[1].set_xlim(0, max(training_results["iteration"]))
    axs[0].minorticks_on()
    axs[1].minorticks_on()
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    plt.title(title, fontsize=22)
    plt.savefig(f"{savedir}/gp_hyperparameters_vs_iteration.png")
    plt.close()


def plot_train_time_vs_iteration(training_results, savedir, title="GP fit"):

    plt.plot(training_results["iteration"], training_results["gp_train_time"], 
                label='GP train step')
    plt.plot(training_results["iteration"], training_results["obj_fn_opt_time"], 
                label='Active learning step')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.title(title, fontsize=22)
    plt.minorticks_on()
    plt.savefig(f"{savedir}/gp_train_time_vs_iteration.png")
    plt.close()


def plot_corner_scatter(tt, yy, labels, savedir, title="GP fit"):

    ndim = theta.shape[1]
    fig = corner.corner(tt, c=yy, labels=labels, 
            plot_datapoints=False, plot_density=False, plot_contours=False,
            show_titles=True, title_kwargs={"fontsize": 18}, 
            label_kwargs={"fontsize": 22})

    axes = np.array(fig.axes).reshape((ndim, ndim))
    cb_rng = [yy.min(), yy.max()]

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            im = ax.scatter(tt.T[xi], tt.T[yi], c=yy, s=2, cmap='coolwarm', 
                            norm=colors.LogNorm(vmin=min(cb_rng), vmax=max(cb_rng)))

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=.98, pad=.1)
    cb.set_label(r'$-\ln P$', fontsize=20)
    cb.set_ticks(cb_rng)
    cb.ax.tick_params(labelsize=18)
    plt.title(title, fontsize=22)
    fig.savefig(f"{savedir}/training_sample_corner.png")
    plt.close()


def plot_gp_fit_1D(theta, y, gp, bounds, title="GP fit"):

    xarr = np.linspace(bounds[0][0], bounds[0][1], 30)
    mu, var = gp.predict(y, xarr, return_var=True)

    fig, ax = plt.subplots()
    plt.plot(xarr, fn(xarr), color='k', linestyle='--', linewidth=.5)
    ax.fill_between(xarr, mu-var, mu+var, color='r', alpha=.8)
    plt.scatter(theta, y, color='r')
    plt.scatter(theta_test, y_test, color='g')
    plt.xlim(bounds[0])
    plt.title(title, fontsize=22)
    plt.savefig(f"{savedir}/gp_fit_1D.png")
    plt.close()


def plot_gp_fit_2D(theta, y, gp, bounds, savedir, ngrid=60, title="GP fit"):

    xarr = np.linspace(bounds[0][0], bounds[0][1], ngrid)
    yarr = np.linspace(bounds[1][0], bounds[1][1], ngrid)

    X, Y = np.meshgrid(xarr, yarr)
    Z = np.zeros((ngrid, ngrid))
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            tt = np.array([X[i][j], Y[i][j]]).reshape(1,-1)
            Z[i][j] = gp.predict(y, tt, return_cov=False)[0]
        
    im = plt.contourf(X, Y, Z, 20, cmap='Blues_r')
    plt.colorbar(im)
    plt.scatter(theta.T[0], theta.T[1], color='r', s=5)
    plt.title(title, fontsize=22)
    plt.savefig(f"{savedir}/gp_fit_2D.png")
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


def plot_emcee_corner(sm):

    fig = corner.corner(sm.emcee_samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        scale_hist=True, plot_contours=True, labels=sm.labels,
                        title_kwargs={"fontsize": 15}, label_kwargs={"fontsize": 15})
    fig.savefig(f"{sm.savedir}/emcee_posterior.png", bbox_inches="tight")


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


def plot_mcmc_comparison(sm):

    lw = 1.5
    colors = ["orange", "royalblue"]

    fig = corner.corner(sm.emcee_samples,  labels=sm.labels, range=self.bounds,
                    show_titles=True, verbose=False, max_n_ticks=4,
                    plot_contours=True, plot_datapoints=True, plot_density=True,
                    color=colors[0], no_fill_contours=False, title_kwargs={"fontsize": 16},
                    label_kwargs={"fontsize": 22}, hist_kwargs={"linewidth":2.0, "density":True})

    fig = corner.corner(sm.dynesty_samples, labels=sm.labels, range=self.bounds, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, verbose=False, max_n_ticks=4, title_fmt='.3f',
                        plot_contours=True, plot_datapoints=True, plot_density=True,
                        color=colors[1], no_fill_contours=False, title_kwargs={"fontsize": 16},
                        label_kwargs={"fontsize": 22}, hist_kwargs={"linewidth":2.0, "density":True},
                        fig=fig)

    fig.axes[1].text(2.2, 0.725, r"--- emcee posterior", fontsize=26, color=colors[0], ha='left')
    fig.axes[1].text(2.2, 0.55, r"--- dynesty posterior", fontsize=26, color=colors[1], ha='left')
    
    fig.savefig(f"{self.savedir}/mcmc_comparison.png")