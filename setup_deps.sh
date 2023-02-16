#!/usr/bin/env bash

# TODO: get repo dir to use it as root folder
# TODO: remove spack termination

# Introduction
echo "Hello!  Thank you for downloading A4MD."
echo
echo "Before we can start installing the software, we'll need to determine which packages you'll need."

a4md_root=$(pwd)

# Verify user has ssh setup
while true; do
	read -p "Do you have an ssh key pair set up in github? (yes/no) " user_has_ssh
	case ${user_has_ssh} in
                [yY] | [yY][eE][sS] ) has_ssh_key="yes"; break ;;
                [nN] | [nN][oO] ) echo "You will need to set up an ssh key with github prior to installation."
			echo "Please review instructions for how to do so at https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent"
			exit ;;
                * ) echo "Please respond with either yes or no: " ;;
	esac
done

# Set up for Environment (C Compiler, OS, and MPI)
while true; do
	read -p "Do you already have a c compiler installed? (yes/no) " user_has_comp
	case ${user_has_comp} in
		[yY] | [yY][eE][sS] ) has_c_comp="yes"; break ;;
		[nN] | [nN][oO] ) has_c_comp="no"
			echo "Please install a version of gcc before continuing"
			exit ;;
		* ) echo "Please respond with either yes or no: " ;;
	esac
done

# Set up for Package Management
while true; do
    read -p "Do you already have the Spack package manager installed? (yes/no) " user_has_spack
    case ${user_has_spack} in
        [yY] | [yY][eE][sS] ) has_spack="yes"; break ;;
        [nN] | [nN][oO] ) echo "You will need to install Spack prior to installation."
            echo "Please review instructions for spack installation at https://spack.readthedocs.io/en/latest/getting_started.html"
            break ;;
                * ) echo "Please respond with either yes or no: " ;;
    esac
done
read -p "Please provide a name for a project spack environment.  It can be anything. (Press enter for default a4md_spack_env) " user_spack_name
while true; do
	read -p "Do you already have the Conda package manager installed? (yes/no) " user_has_conda
	case ${user_has_conda} in
		[yY] | [yY][eE][sS] ) has_conda="yes"; break ;;
		[nN] | [nN][oO] ) echo "You will need to set up use of Anaconda prior to installation."
			echo "Please review instructions for how to do so at https://conda.io/projects/conda/en/latest/user-guide/install/index.html"
			break ;;
                #[nN] | [nN][oO] ) has_conda="no";  read -p "Where would you like to install Conda? (Press enter for your home directory.) " user_conda; break ;;
                * ) echo "Please respond with either yes or no: " ;;
	esac
done

mpi_name="${user_mpi_name:="mpich"}"

# Options of mac, linux86, linuxP9
os_for_conda="${user_os:="linux86"}"

# Needs to be correct path if on machine
#conda_path="${user_conda:=""}"
#spack_path="${user_spack:=""}"

# Can be anything
spack_env_name="${user_spack_name:="a4md_spack_env"}"


#echo ${mpi_name}
cd install
. ./install_a4md_deps.sh ${mpi_name} ${os_for_conda} ${spack_env_name} ${user_has_spack} ${user_has_conda} ${has_c_comp} ${has_ssh_key} ${a4md_root}
cd ..