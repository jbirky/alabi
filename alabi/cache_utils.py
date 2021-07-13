import numpy as np
import pickle
import os

__all__ = ["write_report_gp",
           "write_report_emcee",
           "write_report_dynesty"]


def write_report_gp(self, file):

    # get hyperparameter names and values
    hp_name = self.gp.get_parameter_names()
    hp_vect = self.gp.get_parameter_vector()

    # print model summary to human-readable text file
    lines =  f"==================================================================\n"
    lines += f"GP summary \n"
    lines += f"==================================================================\n\n"

    lines += f"Configuration: \n"
    lines += f"-------------- \n"
    lines += f"Kernel: {self.kernel_name} \n"
    lines += f"Function bounds: {self.bounds} \n"
    lines += f"GP white noise: {self.white_noise} \n"
    lines += f"Active learning algorithm : {self.algorithm} \n\n" 

    lines += f"Number of total training samples: {self.ntrain} \n"
    lines += f"Number of initial training samples: {self.ninit_train} \n"
    lines += f"Number of active training samples: {self.nactive} \n"
    lines += f"Number of test samples: {self.ntest} \n\n"

    lines += f"Results: \n"
    lines += f"-------- \n"
    lines += f"GP final hyperparameters: \n"
    for ii in range(len(hp_name)):
        lines += f"   [{hp_name[ii]}] \t{hp_vect[ii]} \n"
    lines += "\n"

    lines += f"Active learning train runtime (s): {np.round(self.train_runtime)} \n\n"

    lines += f"Final test error: {self.training_results['test_error'][-1]} \n\n"

    summary = open(file+".txt", "w")
    summary.write(lines)
    summary.close()


def write_report_emcee(self, file):

    # get acceptance fraction and autocorrelation time
    acc_frac = np.mean(self.sampler.acceptance_fraction)
    autcorr_time = np.mean(self.sampler.get_autocorr_time())

    # compute summary statistics 
    means = np.mean(self.emcee_samples, axis=0)
    stds = np.std(self.emcee_samples, axis=0)

    lines =  f"==================================================================\n"
    lines += f"emcee summary \n"
    lines += f"==================================================================\n\n"

    lines += f"Configuration: \n"
    lines += f"-------------- \n"
    lines += f"Prior: {self.lnprior_comment} \n\n"

    lines += f"Number of walkers: {self.nwalkers} \n"
    lines += f"Number of steps per walker: {self.nsteps} \n\n"

    lines += f"Results: \n"
    lines += f"-------- \n"
    lines += "Mean acceptance fraction: {0:.3f} \n".format(acc_frac)
    lines += "Mean autocorrelation time: {0:.3f} steps \n".format(autcorr_time)
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
    lines += f"Prior: {self.ptform_comment} \n\n"

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
