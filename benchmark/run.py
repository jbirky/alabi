from alabi.core import *
from alabi.utility import *
from alabi.benchmarks import *
from alabi.visualization import *

import numpy as np
import matplotlib.pyplot as plt
import pickle

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

    kernel = kernels[9]
    benchmark = "eggbox"

    # Plot true function
    plot_true_fit_2D(eval(benchmark)["fn"], eval(benchmark)["bounds"], 
                     savedir=f"results/{benchmark}")

    # Train surrogate model
    sm = SurrogateModel(fn=eval(benchmark)["fn"], 
                        bounds=eval(benchmark)["bounds"], 
                        savedir=f"results/{benchmark}/{kernel}")
    sm.init_samples(ntrain=200, ntest=200)
    sm.init_gp(kernel=kernel, fit_amp=True, white_noise=None)
    sm.active_train(niter=200, algorithm="bape", gp_opt_freq=20)
    sm.plot(plots=["gp_error", "gp_hyperparam", "gp_timing", "gp_fit_2D"])

    # Run MCMC!
    sm.run_emcee(nwalkers=20, nsteps=5e4, opt_init=False)
    sm.plot(plots=["emcee_all"])

    sm.run_dynesty()
    sm.plot(plots=["dynesty_all"])