## Installation

```bash
git clone https://github.com/jbirky/alabi
cd alabi
python setup.py install
```

## Quickstart Example

### Step 1

Import python modules:
```python
import numpy as np
import matplotlib.pyplot as plt

from alabi.core import SurrogateModel
import alabi.utility as ut

np.random.seed(7)
```

### Step 2

Define the test function and the bounds for the input space. For example:
```python
def test1d_fn(x):
    return np.sin(5 * x) * (1 - np.tanh(x**2))

bounds = [(-1, 1)]
```

### Step 3

Initialize the [surrogate model](https://en.wikipedia.org/wiki/Surrogate_model) with a) the function to train on, b) the bounds of the input space, and c) directory where the results will be saved:
```python
sm = SurrogateModel(lnlike_fn=test1d_fn, bounds=bounds, savedir=f"results/test1d")
```

### Step 4

Initialize the Gaussian process (GP) surrogate model by specifying a [kernel](https://en.wikipedia.org/wiki/Kernel_density_estimation). In this example we'll use a squared exponential kernel:

$ k(x, x') = \sigma_f^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right) $

where $ k(x, x') $ is the kernel function, $ \sigma_f^2 $ is the  [hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)'s amplitude, $ \ell $ is the length scale hyperparameter, and $ x $ and $ x' $ are input points.

```python
sm.init_samples(ntrain=10)
```
Then initialize the GP surrogate model on those training points:
```python
sm.init_gp(kernel="ExpSquaredKernel", fit_amp=True, fit_mean=True, fit_white_noise=False)
```

### Step 5

Improve the surrogate model fit by iteratively selecting new training points using [active learning](https://dl.acm.org/doi/10.5555/2832747.2832752)):
```python
sm.active_train(niter=30, algorithm="bape", gp_opt_freq=10)
```

<!-- sphinx-apidoc -o source ../../alabi -->
<!-- sphinx-build -b html source build; make html -->

### Step 6

Run [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) sampler using either the [`emcee`](https://github.com/dfm/emcee) package:
```python
sm.run_emcee(nwalkers=20, nsteps=int(5e4), opt_init=False)
```
or [`dynesty`](https://github.com/joshspeagle/dynesty) nested sampling package:
```python
sm.run_dynesty()
```
Both samplers produce consistent results:
```python
plt.hist(sm.emcee_samples.T[0], bins=50, histtype='step', density=True, label="emcee samples")
plt.hist(sm.dynesty_samples.T[0], bins=50, histtype='step', density=True, label="dynesty samples")
plt.xlabel("$x$", fontsize=25)
plt.legend(loc="upper right", fontsize=18, frameon=False)
plt.minorticks_on()
plt.show()
```