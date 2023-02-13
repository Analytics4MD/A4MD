#!/usr/bin/env bash

install_dir=$1
dataspaces_install_dir=$2

cd ${install_dir}
mkdir build
cd ${install_dir}/build
cmake .. -DCMAKE_INSTALL_PREFIX=${install_dir}/../a4md-test -DDATASPACES_PREFIX=${dataspaces_install_dir}
make
make install
cd ${install_dir}
pip install -e .

