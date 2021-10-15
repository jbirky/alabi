import alabi
from alabi.core import *
from alabi.utility import *
from alabi.benchmarks import *
from alabi.visualization import *

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


if __name__ == "__main__":

    np.random.seed(0)

    kernel = kernels[13]
    benchmark = "eggbox"
    savedir = f"results_/{benchmark}/{kernel}"

    # Plot true function
    # plot_true_fit_2D(eval(benchmark)["fn"], eval(benchmark)["bounds"], 
    #                  savedir=f"results_/{benchmark}")

    sm = SurrogateModel(fn=eval(benchmark)["fn"], 
                        bounds=eval(benchmark)["bounds"], 
                        prior_sampler=None,
                        savedir=savedir,
                        scale="log")
    sm.init_samples(ntrain=150, ntest=150, reload=False)
    sm.init_gp(kernel=kernel, fit_amp=True, fit_mean=True, white_noise=-12)
    sm.active_train(niter=50, algorithm="bape", gp_opt_freq=20)
    sm.plot(plots=["gp_all"])
    breakpoint()

    # gif_frames = []
    # for i in range(50):
    #     sm.active_train(niter=1, algorithm="bape", gp_opt_freq=10)
    #     sm.plot(plots=["gp_fit_2D", "obj_fn_2D", "gp_hyperparam", "gp_error"])
    #     frame = plot_2D_panel4(savedir)
    #     gif_frames.append(frame)
    # imageio.mimsave(f"{savedir}/diff_evol.gif", gif_frames) 