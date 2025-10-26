"""
:py:mod:`cache_utils.py` 
-------------------------------------
"""

import numpy as np
import pickle
import os
from . import parallel_utils

__all__ = ["load_pickle",
           "load_model_cache",
           "write_report_gp",
           "write_report_emcee",
           "write_report_dynesty"]


def load_pickle(savedir, fname="surrogate_model.pkl"):

    file = os.path.join(savedir, fname)
    with open(file, "rb") as f:
        sm = pickle.load(f)

    return sm


def load_model_cache(savedir):
    """
    MPI-safe model loading that prevents file corruption.
    
    :param savedir: Directory containing the model cache
    :returns: Loaded surrogate model
    """
    
    # Check if we're in an MPI environment
    if parallel_utils.is_mpi_active():
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        except ImportError:
            rank = 0
    else:
        rank = 0
    
    # Only rank 0 loads the model
    if rank == 0:
        try:
            sm = load_pickle(savedir)
        except Exception as e:
            print(f"Rank {rank}: Failed to load model cache: {e}")
            raise  # Re-raise the exception to properly handle the error

    else:
        sm = load_pickle(savedir)
    
    # Broadcast the model to all ranks if using MPI
    if parallel_utils.is_mpi_active():
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            sm = comm.bcast(sm, root=0)
            if rank != 0:
                print(f"Rank {rank}: Received model from rank 0")
        except ImportError:
            pass
    
    return sm


def write_report_gp(self, file):

    # get hyperparameter names and values
    hp_name = self.gp.get_parameter_names()
    hp_vect = self.gp.get_parameter_vector()

    # print model summary to human-readable text file
    lines =  f"==================================================================\n"
    lines += f"GP summary \n"
    lines += f"==================================================================\n\n"
    
    report_vars = {"Kernel": "kernel_name",
                   "Function bounds": "bounds",
                   "fit mean": "fit_mean",
                   "fit amplitude": "fit_amp",
                   "fit white_noise": "fit_white_noise",
                   "GP white noise": "white_noise",
                   "Hyperparameter bounds": "hp_bounds",
                   "Active learning algorithm": "algorithm",
                   "Number of total training samples": "ntrain",
                   "Number of initial training samples": "ninit_train",
                   "Number of active training samples": "nactive",
                   "Number of test samples": "ntest",
    }
    
    lines += f"Configuration: \n"
    lines += f"-------------- \n"
    for key in report_vars.keys():
        if hasattr(self, report_vars[key]):
            lines += f"{key}: {getattr(self, report_vars[key])} \n"
    lines += "\n"

    lines += f"Results: \n"
    lines += f"-------- \n"
    lines += f"GP final hyperparameters: \n"
    for ii in range(len(hp_name)):
        lines += f"   [{hp_name[ii]}] \t{hp_vect[ii]} \n"
    lines += "\n"

    if hasattr(self, 'train_runtime'):
        lines += f"Active learning train runtime (s): {np.round(self.train_runtime)} \n\n"

    if hasattr(self, 'training_results'):
        lines += f"Final test error (MSE): {self.training_results['test_mse'][-1]} \n\n"

    summary = open(file+".txt", "w")
    summary.write(lines)
    summary.close()


def write_report_emcee(self, file):

    # compute summary statistics 
    means = np.mean(self.emcee_samples, axis=0)
    stds = np.std(self.emcee_samples, axis=0)

    lines =  f"==================================================================\n"
    lines += f"emcee summary \n"
    lines += f"==================================================================\n\n"

    lines += f"Configuration: \n"
    lines += f"-------------- \n"

    lines += f"Number of walkers: {self.nwalkers} \n"
    lines += f"Number of steps per walker: {self.nsteps} \n\n"

    lines += f"Results: \n"
    lines += f"-------- \n"
    lines += "Mean acceptance fraction: {0:.3f} \n".format(self.acc_frac)
    lines += "Mean autocorrelation time: {0:.3f} steps \n".format(self.autcorr_time)
    lines += f"Burn: {self.iburn} \n"
    lines += f"Thin: {self.ithin} \n"
    lines += f"Total burned, thinned, flattened samples: {self.emcee_samples.shape[0]} \n\n"

    lines += f"emcee runtime (s): {np.round(self.emcee_runtime)} \n\n"

    lines += f"Summary statistics: \n"
    for ii in range(self.ndim):
        lines += f"{self.labels[ii]} = {means[ii]} +/- {stds[ii]} \n"
    lines += "\n"

    summary = open(file+".txt", "a")
    summary.write(lines)
    summary.close()


def write_report_dynesty(self, file):

    # compute summary statistics 
    means = np.mean(self.dynesty_samples, axis=0)
    stds = np.std(self.dynesty_samples, axis=0)

    lines =  f"==================================================================\n"
    lines += f"dynesty summary \n"
    lines += f"==================================================================\n\n"

    lines += f"Configuration: \n"
    lines += f"-------------- \n"

    lines += f"Results: \n"
    lines += f"-------- \n"
    lines += f"Total weighted samples: {self.dynesty_samples.shape[0]} \n\n"

    lines += f"Dynesty runtime (s): {np.round(self.dynesty_runtime)} \n\n"

    lines += f"Summary statistics: \n"
    for ii in range(self.ndim):
        lines += f"{self.labels[ii]} = {means[ii]} +/- {stds[ii]} \n"
    lines += "\n"

    summary = open(file+".txt", "a")
    summary.write(lines)
    summary.close()
