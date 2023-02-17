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
dataspaces_install_dir="$HOME/dataspaces"
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

# if [ ${has_spack} = "no" ]; then
#     echo "Cloning and Activating Spack"
#     echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
#     # cd ${spack_path}
#     git clone https://github.com/spack/spack.git ~/spack
#     cd ~/spack
#     git checkout releases/v0.18
#     cd -
#     echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# fi

# # Add Spack to bashrc
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# echo ". ~/spack/share/spack/setup-env.sh" >> ~/.bashrc
# . ~/spack/share/spack/setup-env.sh
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# echo "Done Preparing Spack"
# echo


# # Install zlib
# echo
# echo "Set up and Activate Spack Environment"
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# path_to_spack=$(which spack)
# spack_root=${path_to_spack%/bin/spack}
# . ${spack_root}/share/spack/setup-env.sh
# spack install zlib
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# # Create and activate spack environment
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# spack env create ${spack_env} # ./a4md_env.lock
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# spack env activate ${spack_env}
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# echo "Done Activating Spack Environment"
# echo

# Spack concretize
# echo
# echo "Concretize and Install Spack Packages"
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# spack concretize
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# # Spack install
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# spack install


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


# echo
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# echo "Building a4md..." >> ${a4md_root}/log.installing_a4md_process
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# echo
# . ./download_a4md.sh ${install_dir}
# echo 
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# echo "Done building a4md." >> ${a4md_root}/log.installing_a4md_process
# echo ${progress_delimiter} >> ${a4md_root}/log.installing_a4md_process
# echo


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

