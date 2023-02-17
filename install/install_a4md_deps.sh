#!/usr/bin/env bash


# Take Input Variables
user_mpi_name=$1
user_os=$2
user_spack_name=$3
a4md_root=$8


has_spack=${4:-"yes"}
has_conda=${5:-"yes"}
has_c_comp=${6:-"yes"}
has_ssh=${7:-"yes"}

# Installation dirs
app_install_dir="${a4md_root}/app"
mkdir -p ${app_install_dir}




# Define Conditionals for Installs
mpi_name="${user_mpi_name:="mpich"}"
spack_env="${user_spack_name:="a4md_spack_env"}"
conda_path="${user_conda:=""}"
spack_path="${user_spack:=".."}"
os_for_conda="${user_os:="linux86"}"

### Create Delimiter and Workflow Variables
n_columns=$(stty size | awk '{print $2}')
progress_delimiter=""
for i in `seq 1 ${n_columns}`;
do
    progress_delimiter+="-"
done

#   Don't find compilers here.  If user has specific c compiler, tell them to call spack compiler find prior to making environment and then update compilers.yaml file.

echo
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Building spack..." >> ${a4md_root}/log.installing_a4md_process
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo
. ./install_spack_packages.sh ${app_install_dir}
echo 
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Done building spack." >> ${a4md_root}/log.installing_a4md_process
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo


echo
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Building conda..." >> ${a4md_root}/log.installing_a4md_process
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo
. ./install_conda.sh ${app_install_dir} ${os_for_conda}
echo 
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Done building conda." >> ${a4md_root}/log.installing_a4md_process
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo

echo
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Building dataspaces..." >> ${a4md_root}/log.installing_a4md_process
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo
. ./install_dataspaces.sh ${app_install_dir} ${a4md_root}
echo 
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Done building dataspaces." >> ${a4md_root}/log.installing_a4md_process
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo 


echo
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Building a4md..." >> ${a4md_root}/log.installing_a4md_process
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo
. ./install_a4md.sh ${a4md_root} ${app_install_dir}
echo 
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Done building a4md." >> ${a4md_root}/log.installing_a4md_process
echo $(pwd)
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo

echo
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Building sample workflow..." >> ${a4md_root}/log.installing_a4md_process
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo
. ./install_sample_ensemble.sh ${a4md_root} ${app_install_dir}
echo 
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo "Done building sample workflow." >> ${a4md_root}/log.installing_a4md_process
echo $(pwd)
echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
echo