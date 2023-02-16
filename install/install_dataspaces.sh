#!/usr/bin/env bash
# TODO: create dataspaces folder different than in home directory
dataspaces_install_dir=$1
install_dir=$2
mkdir -p ${dataspaces_install_dir}
echo 'Building DSpaces' 
# Build and install dataspaces into $HOME/dataspaces
cd ${install_dir}/src/a4md/extern/dataspaces
./autogen.sh
CC=$(which mpicc) CXX=$(which mpicxx) FC=$(which mpifort) CFLAGS=-fPIC ./configure --enable-shmem --enable-dart-tcp --prefix=${dataspaces_install_dir}
make
make install
cd -

