import numpy as np
import pickle
import os


def write_report_gp(self, file):

    # print model summary to human-readable text file
    lines =  f"==========================================\n"
    lines += f"GP summary \n"
    lines += f"==========================================\n\n"

    lines += f"Kernel: {self.kernel_name} \n"
    lines += f"Function bounds: {self.bounds} \n"
    lines += f"Active learning algorithm : {self.algorithm} \n\n" 

    lines += f"GP hyperparameter names: {self.gp.get_parameter_names()} \n"
    lines += f"GP hyperparameter values (final): {self.gp.get_parameter_vector()} \n"
    lines += f"GP hyperparameter optimization frequency: {self.gp_opt_freq} \n\n"

    lines += f"Number of total training samples: {self.ntrain} \n"
    lines += f"Number of initial training samples: {self.ninit_train} \n"
    lines += f"Number of active training samples: {self.nactive} \n"
    lines += f"Number of test samples: {self.ntest} \n"

    lines += f"Final test error: {self.training_results['test_error'][-1]} \n\n"

    summary = open(file+".txt", "w")
    summary.write(lines)
    summary.close()


def write_report_emcee(self, file):

    # get acceptance fraction and autocorrelation time
    acc_frac = np.mean(self.sampler.acceptance_fraction)
    autcorr_time = np.mean(self.sampler.get_autocorr_time())

    lines =  f"==========================================\n"
    lines += f"emcee summary \n"
    lines += f"==========================================\n\n"

    lines += f"Number of walkers: {self.nwalkers} \n"
    lines += f"Number of steps: {self.nsteps} \n\n"

    lines += "Mean acceptance fraction: {0:.3f} \n".format(acc_frac)
    lines += "Mean autocorrelation time: {0:.3f} steps \n".format(autcorr_time)
    lines += f"Burn: {self.iburn} \n"
    lines += f"Thin: {self.ithin} \n"
    lines += f"Total burned, thinned, flattened samples: {self.emcee_samples.shape[0]} \n\n"

    summary = open(file+".txt", "a")
    summary.write(lines)
    summary.close()