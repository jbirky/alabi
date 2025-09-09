# Stellar Evolution

In this example, we show how to use alabi with a model that is relatively slow (~20 seconds per model evaluation). 

To run this example, you'll first need to install the packages `vplanet` and `vplanet_inference`.

Install [vplanet](https://github.com/VirtualPlanetaryLaboratory/vplanet):
```bash
python -m pip install vplanet
```

Install [vplanet_inference](https://github.com/jbirky/vplanet_inference):
```bash
git clone https://github.com/jbirky/vplanet_inference
cd vplanet_inference
python setup.py install
```

### Python script:
```python
import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import astropy.units as u
import numpy as np
from functools import partial
import scipy
import os


# ========================================================
# Configure vplanet forward model
# ========================================================

inpath = os.path.join(vpi.INFILE_DIR, "stellar")

inparams  = {"star.dMass": u.Msun,          
             "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),   
             "star.dSatXUVTime": u.Gyr,    
             "vpl.dStopTime": u.Gyr,       
             "star.dXUVBeta": -u.dimensionless_unscaled}

outparams = {"final.star.Luminosity": u.Lsun,
             "final.star.LXUVStellar": u.Lsun}

vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams)

# ========================================================
# Observational constraints
# ========================================================

# Data: (mean, stdev)
prior_data = [(None, None),     # mass [Msun]
              (-2.92, 0.26),    # log(fsat) 
              (None, None),     # tsat [Gyr]
              (7.6, 2.2),       # age [Gyr]
              (-1.18, 0.31)]    # beta

like_data = np.array([[5.22e-4, 0.19e-4],   # Lbol [Lsun]
                      [7.5e-4, 1.5e-4]])    # Lxuv/Lbol

# Prior bounds
bounds = [(0.07, 0.11),        
          (-5.0, -1.0),
          (0.1, 12.0),
          (0.1, 12.0),
          (-2.0, 0.0)]

# ========================================================
# Configure prior 
# ========================================================

# Prior sampler - alabi format
ps = partial(ut.prior_sampler_normal, prior_data=prior_data, bounds=bounds)

# Prior - emcee format
lnprior = partial(ut.lnprior_normal, bounds=bounds, data=prior_data)

# Prior - dynesty format
prior_transform = partial(ut.prior_transform_normal, bounds=bounds, data=prior_data)

# ========================================================
# Configure likelihood
# ========================================================

# vpm.initialize_bayes(data=like_data, bounds=bounds, outparams=outparams)

def lnlike(theta):
    out = vpm.run_model(theta)
    mdl = np.array([out[0], out[1]/out[0]])
    lnl = -0.5 * np.sum(((mdl - like_data.T[0])/like_data.T[1])**2)
    return lnl

def lnpost(theta):
    return lnlike(theta) + lnprior(theta)

# ========================================================
# Run alabi
# ========================================================

kernel = "ExpSquaredKernel"

labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
          r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

sm = SurrogateModel(fn=lnpost, bounds=bounds, prior_sampler=ps, 
                    savedir=f"results/{kernel}", labels=labels, scale="nlog")
sm.init_samples(ntrain=100, ntest=100, reload=False)
sm.init_gp(kernel=kernel, fit_amp=False, fit_mean=True, white_noise=-15)
sm.active_train(niter=500, algorithm="bape", gp_opt_freq=10)
sm.plot(plots=["gp_all"])

sm = alabi.cache_utils.load_model_cache(f"results/{kernel}/")

sm.run_emcee(lnprior=lnprior, nwalkers=50, nsteps=5e4, opt_init=False)
sm.plot(plots=["emcee_corner"])

sm.run_dynesty(ptform=prior_transform, mode='dynamic')
sm.plot(plots=["dynesty_all"])
```