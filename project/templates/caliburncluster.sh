{% extends "slurm.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name="{{ id }}"
#SBATCH --qos main-generic
#SBATCH -A parashar-003

{% if partition %}
#SBATCH --partition={{ partition }}
{% endif %}
{% if walltime %}
#SBATCH -t {{ walltime|format_timedelta }}
{% endif %}
{% if job_output %}
#SBATCH --output={{ job_output }}
#SBATCH --error={{ job_output }}
{% endif %}
{% block tasks %}
SBATCH --ntasks={{ np_global }}
{% endblock %}
module purge
module load python/3.6.3
#module load openmpi/2.1.3-gcc-4.8.5
#module load boost/1.68-gcc-4.8.5
module load openmpi/2.1.3-gcc-8.1.0
module load boost/1.68-gcc-8.1.0

source /home1/st18003/scratch/a4md_env/bin/activate

################## USER PATHS GO HERE ##########################
export PATH="/home1/st18003/scratch/software/install/lammps:$PATH"
export PATH="/home1/st18003/scratch/projects/gromacs-2018.3/_install/bin/:$PATH"
export PATH="/home1/st18003/scratch/projects/a4md/_install/bin:$PATH"

#export TAU_VERBOSE=1
#export TAU_TRACK_SIGNALS=1
#export TAU_METRICS=TIME,PAPI_NATIVE_powercap:::ENERGY_UJ:ZONE0
################################################################
{% endblock %}

{% block body %}
{% set cmd_suffix = cmd_suffix|default('') ~ (' &' if parallel else '') %}
{% for operation in operations %}
{% if operation.directives.nranks and not mpi_prefix %}
{% set mpi_prefix = "%s -n %d "|format(mpiexec|default("mpirun"), operation.directives.nranks) %}
{% endif %}

# {{ "%s"|format(operation) }}
{% if operation.directives.omp_num_threads %}
export OMP_NUM_THREADS={{ operation.directives.omp_num_threads }}
{% endif %}
{{ mpi_prefix }}{{ cmd_prefix }}{{ operation.cmd }}{{ cmd_suffix }}
{% endfor %}
{% endblock %}

