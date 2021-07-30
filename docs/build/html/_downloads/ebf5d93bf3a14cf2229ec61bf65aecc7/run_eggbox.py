"""
Eggbox (2D)
===========
"""

# %%
# Initialize training function
# ----------------------------

from alabi.core import SurrogateModel
from alabi.benchmarks import eggbox
import alabi.visualization as vis

kernel = "Matern52Kernel"
benchmark = "eggbox"

vis.plot_true_fit_2D(eval(benchmark)["fn"], eval(benchmark)["bounds"], 
                     savedir=f"results/{benchmark}")

# %%
# .. admonition:: results/eggbox/true_function_2D.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/eggbox/true_function_2D.png


# %%
# Initialize GP surrogate model
# ----------------------------

sm = SurrogateModel(fn=eval(benchmark)["fn"], 
                    bounds=eval(benchmark)["bounds"], 
                    savedir=f"results/{benchmark}/{kernel}")

# %%
# Train GP surrogate model
# ----------------------------

sm.init_samples(ntrain=200, ntest=200)
sm.init_gp(kernel=kernel, fit_amp=True, fit_mean=True, white_noise=None)
sm.active_train(niter=200, algorithm="bape", gp_opt_freq=20)


# %%
# Plot GP diagnostics
# ----------------------------

sm.plot(plots=["gp_error", "gp_hyperparam", "gp_timing", "gp_fit_2D"])

# %%
# .. admonition:: results/eggbox/Matern52Kernel/gp_error_vs_iteration.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/eggbox/Matern52Kernel/gp_error_vs_iteration.png

# %%
# .. admonition:: results/eggbox/Matern52Kernel/gp_hyperparameters_vs_iteration.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/eggbox/Matern52Kernel/gp_hyperparameters_vs_iteration.png

# %%
# .. admonition:: results/eggbox/Matern52Kernel/gp_train_time_vs_iteration.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/eggbox/Matern52Kernel/gp_train_time_vs_iteration.png

# %%
# .. admonition:: results/eggbox/Matern52Kernel/gp_fit_2D.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/eggbox/Matern52Kernel/gp_fit_2D.png


# %%
# Run MCMC using ``emcee``
# ----------------------------
#
# .. error:: 
# 
# 	 While you can attempt to run ``emcee`` on this surrogate model using ``sm.run_emcee()``, 
# 	 you're likely to find that it won't converge due to ``emcee``'s affine-invariant sampling 
# 	 algorithm failing to explore the multimodal parameter space.


# %%
# Run MCMC using ``dynesty``
# ----------------------------

sm.run_dynesty()
sm.plot(plots=["dynesty_all"])

# %%
# .. admonition:: results/eggbox/Matern52Kernel/dynesty_posterior_kde.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/eggbox/Matern52Kernel/dynesty_posterior_kde.png
