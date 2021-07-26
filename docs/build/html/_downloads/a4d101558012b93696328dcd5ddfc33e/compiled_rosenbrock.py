"""
Rosenbrock
==========
"""

# %%
# Initialize training function
# ----------------------------

from alabi.core import SurrogateModel
from alabi.benchmarks import rosenbrock
import alabi.visualization as vis

kernel = "ExpSquaredKernel"
benchmark = "rosenbrock"

vis.plot_true_fit_2D(eval(benchmark)["fn"], eval(benchmark)["bounds"], 
                     savedir=f"results/{benchmark}")

# %%
# .. admonition:: results/rosenbrock/true_function_2D.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/rosenbrock/true_function_2D.png
#
# .. |docstring| replace:: """


# %%
# Initialize GP surrogate model
# ----------------------------

sm = SurrogateModel(fn=eval(benchmark)["fn"], 
                    bounds=eval(benchmark)["bounds"], 
                    savedir=f"results/{benchmark}/{kernel}")

# %%
# Train GP surrogate model
# ----------------------------

sm.init_samples(ntrain=50, ntest=50)
sm.init_gp(kernel=kernel, fit_amp=True, fit_mean=True, white_noise=None)
sm.active_train(niter=100, algorithm="bape", gp_opt_freq=20)


# %%
# Plot GP diagnostics
# ----------------------------

sm.plot(plots=["gp_error", "gp_hyperparam", "gp_timing", "gp_fit_2D"])

# %%
# .. admonition:: results/rosenbrock/ExpSquaredKernel/gp_error_vs_iteration.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/rosenbrock/ExpSquaredKernel/gp_error_vs_iteration.png
#
# .. |docstring| replace:: """

# %%
# .. admonition:: results/rosenbrock/ExpSquaredKernel/gp_hyperparameters_vs_iteration.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/rosenbrock/ExpSquaredKernel/gp_hyperparameters_vs_iteration.png
#
# .. |docstring| replace:: """

# %%
# .. admonition:: results/rosenbrock/ExpSquaredKernel/gp_train_time_vs_iteration.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/rosenbrock/ExpSquaredKernel/gp_train_time_vs_iteration.png
#
# .. |docstring| replace:: """

# %%
# .. admonition:: results/rosenbrock/ExpSquaredKernel/gp_fit_2D.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/rosenbrock/ExpSquaredKernel/gp_fit_2D.png
#
# .. |docstring| replace:: """


# %%
# Run MCMC using ``emcee``
# ----------------------------

sm.run_emcee(nwalkers=20, nsteps=5e4, opt_init=False)
sm.plot(plots=["emcee_all"])


# %%
# .. admonition:: results/rosenbrock/ExpSquaredKernel/emcee_posterior.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/rosenbrock/ExpSquaredKernel/emcee_posterior.png
#
# .. |docstring| replace:: """


# %%
# Run MCMC using ``dynesty``
# ----------------------------

sm.run_dynesty()
sm.plot(plots=["dynesty_all"])

# %%
# .. admonition:: results/rosenbrock/ExpSquaredKernel/dynesty_posterior.png
#    :class: dropdown, tip
# 
#    .. image:: ../../examples/results/rosenbrock/ExpSquaredKernel/dynesty_posterior.png
#
# .. |docstring| replace:: """
