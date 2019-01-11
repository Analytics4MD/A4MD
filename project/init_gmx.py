import copy
import sys
import signac
import itertools
from collections import OrderedDict


def get_parameters():
    parameters = OrderedDict()
    # Generate Parameters
    parameters["NPROCS"] = [1]
    parameters["T"] = [300]
    parameters["P"] = [1]
    parameters["simulation_time"] = [2000]
    #parameters["L"] = [15]#, 30, 60]
    parameters["stride"] = [10,100]#[10,100,160,200,250,400,500,800,1000]#[1, 10, 20, 100]#, 5000, 10000]#, 100, 500, 1000, 5000, 10000]#[10, 50, 100, 500, 1000, 5000]#, 10000, 20000]
    parameters["trial"] = [1]#, 2, 3, 4, 5]
    parameters['filter_group'] = [('Protein_NA_bound',21)]
    parameters["job_type"] = ['plumed_ds_concurrent','plumed_ds_sequential','plumed_sequential','traditional']
    parameters["output_type"] = ['xtc']#'xyz','dcd']
    parameters["simulation_engine"] = ['gromacs']
    parameters['tau_profiling'] = [True]
    return list(parameters.keys()), list(itertools.product(*parameters.values()))


if __name__ == "__main__":
    project = signac.init_project('a4md')
    gen_param_names, gen_param_combinations = get_parameters()
    # Create the generate jobs
    for gen_params in gen_param_combinations:
        statepoint = dict(zip(gen_param_names, gen_params))
        job = project.open_job(statepoint)
        job.init()
    project.write_statepoints()

