import copy
import sys
import signac
import itertools
from collections import OrderedDict


def get_parameters():
    parameters = OrderedDict()
    # Generate Parameters
    parameters["NPROCS"] = [1]
    parameters["T"] = [1]
    parameters["L"] = [15] #, 30, 60]
    parameters["data_dump_interval"] = [1000, 5000, 10000, 20000]
    parameters["trials"] = [1, 2, 3, 4, 5]
    parameters["job_type"] = ['traditional','plumed_sequential']
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

