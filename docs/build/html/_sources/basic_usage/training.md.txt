# GP Training

```{warning} 
Page under construction
````

## Initializing Training Samples

```{note} 
* 'test' samples are used to assess GP fit
* for numerical stability training data are scaled (theta_min, theta_max) -> (0,1) and (y_mean, y_std) -> (0,1)
* by default this function will run in parallel utilizing all cores unless otherwise specified
* by default this function will choose samples using the 'sobol' hypercube sampling method
````

## Initializing the GP

```{note} 
Choosing a Kernel
````

## Active Learning Training

Algorithms:
* BAPE - Kandasamy et al. (2017)
* AGP - Wang & Li (2017)
* Jones - Jones et al. (1998)


## Evaluating GP Perfomance 

```{note} 
Diagnostic plots and how to interpret
````


## Caching Results

By default the model will cache the `SurrogateModel` object to a pickle file called `surrogate_model.pkl` and will output a summary text file `surrogate_model.txt` looking something like this:

```
==================================================================
GP summary 
==================================================================

Configuration: 
-------------- 
Kernel: ExpSquaredKernel 
Function bounds: [(-5, 5), (-5, 5)] 
GP white noise: None 
Active learning algorithm : bape 

Number of total training samples: 150 
Number of initial training samples: 50 
Number of active training samples: 100 
Number of test samples: 50 

Results: 
-------- 
GP final hyperparameters: 
   [mean:value] 	-1.4285555081915942 
   [kernel:k1:log_constant] 	11.974832220353308 
   [kernel:k2:metric:log_M_0_0] 	0.28389110292099495 
   [kernel:k2:metric:log_M_1_1] 	6.953601851332223 

Active learning train runtime (s): 27.0 

Final test error: 8.128303113409325e-08 
```

To reload the surrogate model object again to a script you can use:
```python
from alabi.cache_utils import load_model_cache

sm = load_model_cache(f"surrogate_model.pkl")
```