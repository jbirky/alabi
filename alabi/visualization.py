import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)

__all__ = ["plot_error_vs_iteration", 
           "plot_hyperparam_vs_iteration", 
           "plot_train_time_vs_iteration",
           "plot_corner_scatter",
           "plot_gp_fit_1D",
           "plot_gp_fit_2D"]


def plot_error_vs_iteration(training_results, savedir, log=False):

    plt.plot(training_results["iteration"], training_results["training_error"], 
                label='train error')
    plt.plot(training_results["iteration"], training_results["test_error"], 
                label='test error')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    plt.legend(loc='best')
    plt.minorticks_on()
    plt.tight_layout()
    if log:
        plt.yscale('log')
        plt.savefig(f"{savedir}/gp_error_vs_iteration_log.png")
    else:
        plt.savefig(f"{savedir}/gp_error_vs_iteration.png")
    plt.close()


def plot_hyperparam_vs_iteration(training_results, hp_names, 
                                 hp_values, savedir):

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
    plt.tight_layout()
    plt.savefig(f"{savedir}/gp_hyperparameters_vs_iteration.png")
    plt.close()


def plot_train_time_vs_iteration(training_results, savedir):

    plt.plot(training_results["iteration"], training_results["gp_train_time"], 
                label='GP train time')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.legend(loc='best')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(f"{savedir}/gp_train_time_vs_iteration.png")
    plt.close()


def plot_corner_scatter(tt, yy, labels, savedir):

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
    fig.savefig(f"{savedir}/training_sample_corner.png")
    plt.close()


def plot_gp_fit_1D(theta, y, gp, bounds):

    xarr = np.linspace(bounds[0][0], bounds[0][1], 30)
    mu, var = gp.predict(y, xarr, return_var=True)

    fig, ax = plt.subplots()
    plt.plot(xarr, fn(xarr), color='k', linestyle='--', linewidth=.5)
    ax.fill_between(xarr, mu-var, mu+var, color='r', alpha=.8)
    plt.scatter(theta, y, color='r')
    plt.scatter(theta_test, y_test, color='g')
    plt.xlim(bounds[0])
    plt.savefig(f"{savedir}/gp_fit_1D.png")
    plt.close()


def plot_gp_fit_2D(theta, y, gp, bounds, savedir, ngrid=60):

    xarr = np.linspace(bounds[0][0], bounds[0][1], ngrid)
    yarr = np.linspace(bounds[1][0], bounds[1][1], ngrid)

    X, Y = np.meshgrid(xarr, yarr)
    Z = np.zeros((ngrid, ngrid))
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            tt = np.array([X[i][j], Y[i][j]]).reshape(1,-1)
            Z[i][j] = gp.predict(y, tt, return_cov=False)[0]
        
    plt.contourf(X, Y, Z, 20, cmap='gist_heat')
    plt.colorbar()
    plt.savefig(f"{savedir}/gp_fit_2D.png")
    plt.close()