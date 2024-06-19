import alabi
from alabi.core import *
from alabi.utility import *
from alabi.benchmarks import *
from alabi.visualization import *
from alabi.cache_utils import load_model_cache

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pickle
import time
import imageio

kernels  = {0: 'Kernel',
            1: 'Sum',
            2: 'Product',
            3: 'ConstantKernel',
            4: 'CosineKernel',
            5: 'DotProductKernel',
            6: 'EmptyKernel',
            7: 'ExpKernel',
            8: 'ExpSine2Kernel',
            9: 'ExpSquaredKernel',
            10: 'LinearKernel',
            11: 'LocalGaussianKernel',
            12: 'Matern32Kernel',
            13: 'Matern52Kernel',
            14: 'MyLocalGaussianKernel',
            15: 'PolynomialKernel',
            16: 'RationalQuadraticKernel'}

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white"})


def test1d_fn(theta):

    theta = np.asarray(theta)
    return -np.sin(3*theta) + np.cos(2*theta-2) * theta**2 + 0.7*theta 

test1d_bounds = [(-2,3)]

testfn = {"fn": test1d_fn,
          "bounds": test1d_bounds}


if __name__ == "__main__":

    # np.random.seed(0)

    kernel = kernels[9]
    benchmark = "testfn"
    savedir = f"results_/{benchmark}/{kernel}"

    # Plot true function
    # plot_true_fit_2D(eval(benchmark)["fn"], eval(benchmark)["bounds"], 
    #                  savedir=f"results_/{benchmark}")

    sm = SurrogateModel(fn=eval(benchmark)["fn"], 
                        bounds=eval(benchmark)["bounds"], 
                        prior_sampler=None,
                        savedir=savedir)
    sm.init_samples(ntrain=10, ntest=0, reload=False)
    sm.init_gp(kernel=kernel, fit_amp=True, fit_mean=True, white_noise=-12)
    sm.active_train(niter=10, algorithm="bape", gp_opt_freq=20, save_progress=False)
    sm.plot(plots=["gp_all"])

    # for ii in range(20):
    #     sm.active_train(niter=1, algorithm="bape", gp_opt_freq=20, save_progress=False)
    #     xarr = np.linspace(sm.bounds[0][0], sm.bounds[0][1], 80)
    #     mu, var = sm.gp.predict(sm.y, xarr, return_var=True)

    #     fig, ax = plt.subplots(figsize=[20,10])
    #     # plt.plot(xarr, sm.fn(xarr), color='w', linestyle='--', linewidth=.5, label="True Model")
    #     # ax.fill_between(xarr, mu-var, mu+var, color='r', alpha=.6, label="Surrogate Model")
    #     plt.plot(xarr, sm.fn(xarr), color='w', linestyle='--', linewidth=3)
    #     ax.fill_between(xarr, mu-var, mu+var, color='r', alpha=.6)
    #     plt.scatter(sm.theta, sm.y, color='r')
    #     plt.scatter(sm.theta_test, sm.y_test, color='g')
    #     plt.xlim(sm.bounds[0])
    #     plt.ylim(-7,5)
    #     # plt.title("Surrogate Model", fontsize=22)
    #     # plt.legend(loc='lower right', frameon=False)
    #     plt.xlabel(r"$\theta$", fontsize=20)
    #     ax.axes.yaxis.set_visible(False)
    #     ax.axes.xaxis.set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     plt.ylabel(r"$\rm Posterior P(\theta)$", fontsize=20)
    #     plt.tight_layout()
    #     plt.style.use('dark_background')
    #     plt.savefig(f"{sm.savedir}/animation/frame{ii}.png", transparent=True)
    #     plt.close()

    # sm = load_model_cache(savedir)
    sm.run_dynesty(like_fn="true")
    # sm.plot(plots=["dynesty_all"])

    # gif_frames = []
    # for i in range(50):
    #     sm.active_train(niter=1, algorithm="bape", gp_opt_freq=10)
    #     sm.plot(plots=["gp_fit_2D", "obj_fn_2D", "gp_hyperparam", "gp_error"])
    #     frame = plot_2D_panel4(savedir)
    #     gif_frames.append(frame)
    # imageio.mimsave(f"{savedir}/diff_evol.gif", gif_frames) 