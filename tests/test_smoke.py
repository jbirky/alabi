"""
Smoke tests for alabi. These verify that the package imports correctly and
that core functionality works without running expensive GP training.
"""

import numpy as np
import pytest


# ===========================================================================
# Import tests
# ===========================================================================

def test_import_alabi():
    import alabi


def test_import_submodules():
    from alabi import benchmarks, core, gp_utils, metrics, utility, visualization


def test_surrogate_model_accessible():
    from alabi.core import SurrogateModel


# ===========================================================================
# Benchmark functions
# ===========================================================================

def test_test1d():
    from alabi.benchmarks import test1d
    fn = test1d["fn"]
    bounds = test1d["bounds"]
    assert len(bounds) == 1
    result = fn(np.array([0.0]))
    assert np.isfinite(result)


def test_rosenbrock():
    from alabi.benchmarks import rosenbrock
    fn = rosenbrock["fn"]
    result = fn(np.array([1.0, 1.0]))
    assert np.isfinite(result)


def test_gaussian_2d():
    from alabi.benchmarks import gaussian_2d
    fn = gaussian_2d["fn"]
    result = fn(np.array([0.5, 0.5]))
    assert np.isfinite(result)


def test_gaussian_shells():
    from alabi.benchmarks import gaussian_shells
    fn = gaussian_shells["fn"]
    result = fn(np.array([0.0, 0.0]))
    assert np.isfinite(result)


def test_eggbox():
    from alabi.benchmarks import eggbox
    fn = eggbox["fn"]
    result = fn(np.array([0.5, 0.5]))
    assert np.isfinite(result)


# ===========================================================================
# SurrogateModel instantiation
# ===========================================================================

def test_surrogate_model_init():
    from alabi.core import SurrogateModel
    from alabi.benchmarks import test1d

    sm = SurrogateModel(
        lnlike_fn=test1d["fn"],
        bounds=test1d["bounds"],
    )
    assert sm is not None


def test_surrogate_model_init_samples():
    from alabi.core import SurrogateModel
    from alabi.benchmarks import test1d

    sm = SurrogateModel(
        lnlike_fn=test1d["fn"],
        bounds=test1d["bounds"],
    )
    sm.init_samples(ntrain=5)
    assert len(sm.theta_train) == 5
    assert len(sm.y_train) == 5


def test_surrogate_model_init_gp():
    from alabi.core import SurrogateModel
    from alabi.benchmarks import test1d

    sm = SurrogateModel(
        lnlike_fn=test1d["fn"],
        bounds=test1d["bounds"],
    )
    sm.init_samples(ntrain=5)
    sm.init_gp()
    assert sm.gp is not None


def test_surrogate_model_2d():
    from alabi.core import SurrogateModel
    from alabi.benchmarks import gaussian_2d

    sm = SurrogateModel(
        lnlike_fn=gaussian_2d["fn"],
        bounds=gaussian_2d["bounds"],
    )
    sm.init_samples(ntrain=8)
    sm.init_gp()
    assert sm.gp is not None
