#!/usr/bin/env bash
app_install_dir=$1
cd ${app_install_dir}
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
# TODO: remove path 
# TODO: install conda according the OS
bash Anaconda3-5.3.1-Linux-x86_64.sh -b -p ${app_install_dir}/anaconda3
. ${app_install_dir}/anaconda3/etc/profile.d/conda.sh
rm -rf Anaconda3-5.3.1-Linux-x86_64.sh
conda create -y -n a4md_conda_env numpy scipy
conda activate a4md_conda_env
conda install -y python=3.9.13=h2660328_0_cpython -c conda-forge
conda install -y -c conda-forge tbb tbb-devel cython sympy scikit-build
conda install -y -c conda-forge freud
pip install --upgrade MDAnalysis

export LD_LIBRARY_PATH="~/anaconda3/envs/a4md_conda_env/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="~/anaconda3/envs/a4md_conda_env/lib:${LIBRARY_PATH}"

cd -
