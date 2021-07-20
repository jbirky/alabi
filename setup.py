from setuptools import setup

setup(name="alabi",
      version="0.0.1",
      description="Active Learning Accelerated Bayesian Inference (ALABI)",
      author="Jessica Birky",
      author_email="jbirky@uw.edu",
      license = "MIT",
      url="https://github.com/jbirky/alabi",
      packages=["alabi"],
      install_requires = ["numpy",
                          "matplotlib >= 2.0.0",
                          "scipy",
                          "george",
                          "emcee >= 3.0",
                          "dynesty",
                          "corner",
                          "sklearn",
                          "pybind11",
                          "pytest",
                          "h5py",
                          "tqdm"]
     )