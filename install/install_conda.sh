#!/usr/bin/env bash
app_install_dir=$1
os_for_conda=$2


cd ${app_install_dir}

if [ ${os_for_conda} = "linux86" ]; then
    echo ${progress_delimiter}
    wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
    echo ${progress_delimiter}
    # Install conda with bash script
    echo ${progress_delimiter}
    bash Anaconda3-5.3.1-Linux-x86_64.sh -b -p ${app_install_dir}/anaconda3
    echo ${progress_delimiter}
    rm -rf Anaconda3-5.3.1-Linux-x86_64.sh

fi
if [ ${os_for_conda} = "linuxP9" ]; then
    echo ${progress_delimiter}
    wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-ppc64le.sh
    echo ${progress_delimiter}
    # Install conda with bash script
    echo ${progress_delimiter}
    bash Anaconda3-2022.05-Linux-ppc64le.sh -b -p ${app_install_dir}/anaconda3
    echo ${progress_delimiter}
fi
if [ ${os_for_conda} = "mac" ]; then
    echo ${progress_delimiter}
    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-MacOSX-x86_64.sh
    echo ${progress_delimiter}
    # Install conda with bash script
    echo ${progress_delimiter}
    bash Anaconda3-2022.10-MacOSX-x86_64.sh -b -p ${app_install_dir}/anaconda3
    echo ${progress_delimiter}
fi


. ${app_install_dir}/anaconda3/etc/profile.d/conda.sh

conda create -y -n a4md_conda_env numpy scipy
conda activate a4md_conda_env
conda install -y python=3.9.13=h2660328_0_cpython -c conda-forge
conda install -y -c conda-forge tbb tbb-devel cython sympy scikit-build
conda install -y -c conda-forge freud
pip install --upgrade MDAnalysis

export LD_LIBRARY_PATH="${app_install_dir}/anaconda3/envs/a4md_conda_env/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${app_install_dir}/anaconda3/envs/a4md_conda_env/lib:${LIBRARY_PATH}"
cd -