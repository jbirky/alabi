# MCMC

```{warning} 
Page under construction
````

First, we can load the surrogate model trained in the [GP Training tutorial](https://jbirky.github.io/alabi/build/basic_usage/training.html) that was cached using `pickle`:
```python
from alabi.cache_utils import load_model_cache

sm = load_model_cache(f"surrogate_model.pkl")
```

## ``emcee``

```{note} 
Describe `emcee` 
````

### Running MCMC: Default

To run `emcee` with a uniform prior and default settings you can simply use:
```python
sm.run_emcee()
```


### Configuring Prior Function

By default, if no `lnprior` is specified `sm.run_emcee()` will use a uniform prior using the bounds specified in the `SurrogateModel` initialization. 

#### Non-uniform prior example

Gaussian prior example:

```python
def lnprior(x, bounds=bounds, data=prior_data):

    lnp = alabi.utility.lnprior_uniform(x, bounds)

    for ii in range(len(x)):
        if data[ii][0] is not None:
            lnp += norm.logpdf(x[ii], data[ii][0], data[ii][1])

    return lnp
```

### Running MCMC: Custom


<!-- ================================================================== -->


## `dynesty`

```{note} 
Describe `dynesty` 

### Running MCMC: Default

To run `dynesty` with a uniform prior and default settings you can simply use:
```python
sm.run_dynesty()
```


### Configuring Prior Function

By default, if no `ptform` is specified  `sm.run_dynesty()` will use a uniform prior using the bounds specified in the `SurrogateModel` initialization. 

#### Non-uniform prior example

Gaussian prior example:


```python
def prior_transform(x, bounds=bounds, data=prior_data):

    pt = np.zeros(len(bounds))
    for i, b in enumerate(bounds):
        if data[i][0] is None:
            # uniform prior transform
            pt[i] = (b[1] - b[0]) * x[i] + b[0]
        else:
            # gaussian prior transform
            pt[i] = scipy.stats.norm.ppf(x[i], data[i][0], data[i][1])
    
    return pt
```

### Running MCMC: Custom