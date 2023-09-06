#!/usr/bin/env bash

install_dir=$1
app_install_dir=$2

cd ${install_dir}
mkdir build
cd ${install_dir}/build
cmake .. -DCMAKE_INSTALL_PREFIX=${install_dir}/a4md-test -DDATASPACES_PREFIX=${app_install_dir}/dataspaces
make
make install
cd ${install_dir}
pip install -e .

