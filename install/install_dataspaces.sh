#!/usr/bin/env bash
app_install_dir=$1
install_dir=$2
mkdir -p ${app_install_dir}/dataspaces
echo 'Building DSpaces' 
# Build and install dataspaces into $HOME/dataspaces
cd ${install_dir}/src/a4md/extern/dataspaces
./autogen.sh
CC=$(which mpicc) CXX=$(which mpicxx) FC=$(which mpifort) CFLAGS=-fPIC ./configure --enable-shmem --enable-dart-tcp --prefix=${app_install_dir}/dataspaces
make
make install
cd -

