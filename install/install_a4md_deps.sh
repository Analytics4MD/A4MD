#!/usr/bin/env bash


# Take Input Variables
user_mpi_name=$1
user_os=$2
user_spack_name=$3

has_spack=${4:-"yes"}
has_conda=${5:-"yes"}
has_c_comp=${6:-"yes"}
has_ssh=${7:-"yes"}

# Installation dirs
install_dir="~/test/a4md"
dataspaces_install_dir="~/dataspaces"


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

if [ ${has_spack} = "no" ]; then
    echo "Cloning and Activating Spack"
    echo ${progress_delimiter}
    # cd ${spack_path}
    git clone https://github.com/spack/spack.git ~/spack
    cd ~/spack
    git checkout releases/v0.18
    cd -
    echo ${progress_delimiter}
fi

# Add Spack to bashrc
echo ${progress_delimiter}
echo ". ~/spack/share/spack/setup-env.sh" >> ~/.bashrc
. ~/spack/share/spack/setup-env.sh
echo ${progress_delimiter}
echo "Done Preparing Spack"
echo
fi

# Install zlib
echo
echo "Set up and Activate Spack Environment"
echo ${progress_delimiter}
path_to_spack=$(which spack)
spack_root=${path_to_spack%/bin/spack}
. ${spack_root}/share/spack/setup-env.sh
spack install zlib
echo ${progress_delimiter}
# Create and activate spack environment
echo ${progress_delimiter}
spack env create ${spack_env} # ./a4md_env.lock
echo ${progress_delimiter}
spack env activate ${spack_env}
echo ${progress_delimiter}
echo "Done Activating Spack Environment"
echo

# Spack concretize
# echo
# echo "Concretize and Install Spack Packages"
# echo ${progress_delimiter}
# spack concretize
# echo ${progress_delimiter}
# # Spack install
# echo ${progress_delimiter}
# spack install


echo
echo ${progress_delimiter}
echo "Building Pluto..." >> installing_a4md_process
echo ${progress_delimiter}
echo
./install_spack_packages.sh
echo 
echo ${progress_delimiter}
echo "Done building Pluto." >> installing_a4md_process
echo ${progress_delimiter}
echo


echo
echo ${progress_delimiter}
echo "Building Pluto..." >> installing_a4md_process
echo ${progress_delimiter}
echo
./install_conda.sh
echo 
echo ${progress_delimiter}
echo "Done building Pluto." >> installing_a4md_process
echo ${progress_delimiter}
echo


echo
echo ${progress_delimiter}
echo "Building Pluto..." >> installing_a4md_process
echo ${progress_delimiter}
echo
./download_a4md.sh ${install_dir}
echo 
echo ${progress_delimiter}
echo "Done building Pluto." >> installing_a4md_process
echo ${progress_delimiter}
echo


echo
echo ${progress_delimiter}
echo "Building Pluto..." >> installing_a4md_process
echo ${progress_delimiter}
echo
./install_dataspaces.sh ${dataspaces_install_dir} ${install_dir}
echo 
echo ${progress_delimiter}
echo "Done building Pluto." >> installing_a4md_process
echo ${progress_delimiter}
echo


echo
echo ${progress_delimiter}
echo "Building Pluto..." >> installing_a4md_process
echo ${progress_delimiter}
echo
./install_a4md.sh ${install_dir} ${dataspaces_install_dir}
echo 
echo ${progress_delimiter}
echo "Done building Pluto." >> installing_a4md_process
echo ${progress_delimiter}
echo



# ./install_spack_packages.sh
# ./install_conda.sh
# ./download_a4md.sh ${install_dir}
# ./install_dataspaces.sh ${dataspaces_install_dir} ${install_dir}
# ./install_a4md.sh ${install_dir} ${dataspaces_install_dir}

