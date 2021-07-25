# Basic Usage

## Rosenbrock Function

### Initialize function
```python
from alabi.core import SurrogateModel
from alabi.benchmarks import rosenbrock
import alabi.visualization as vis

kernel = "ExpSquaredKernel"
benchmark = "rosenbrock"

vis.plot_true_fit_2D(eval(benchmark)["fn"], eval(benchmark)["bounds"], 
                     savedir=f"results/{benchmark}")
```
![](../../benchmark/results/rosenbrock/true_function_2D.png)

### Initialize GP
```python
sm = SurrogateModel(fn=eval(benchmark)["fn"], 
                    bounds=eval(benchmark)["bounds"], 
                    savedir=f"results/{benchmark}/{kernel}")
```

### Train the GP surrogate model
```python
sm.init_samples(ntrain=200, ntest=200)
sm.init_gp(kernel=kernel, fit_amp=True, fit_mean=True, white_noise=None)
sm.active_train(niter=200, algorithm="bape", gp_opt_freq=20)
sm.plot(plots=["gp_error", "gp_hyperparam", "gp_timing", "gp_fit_2D"])
```

![](../../benchmark/results/rosenbrock/ExpSquaredKernel/gp_error_vs_iteration.png)
![](../../benchmark/results/rosenbrock/ExpSquaredKernel/gp_hyperparameters_vs_iteration.png)
![](../../benchmark/results/rosenbrock/ExpSquaredKernel/gp_train_time_vs_iteration.png)
![](../../benchmark/results/rosenbrock/ExpSquaredKernel/gp_fit_2D.png)

### Run MCMC using `emcee`
```python
sm.run_emcee(nwalkers=20, nsteps=5e4, opt_init=False)
sm.plot(plots=["emcee_all"])
```
![](../../benchmark/results/rosenbrock/ExpSquaredKernel/emcee_posterior.png)

### Run MCMC using `dynesty` 
```python
sm.run_dynesty()
sm.plot(plots=["dynesty_all"])
```
![](../../benchmark/results/rosenbrock/ExpSquaredKernel/dynesty_posterior.png)