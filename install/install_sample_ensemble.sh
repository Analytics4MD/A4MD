#!/usr/bin/env bash

install_dir=$1
app_install_dir=$2

echo 'Cloning and building sample workflow' 
cd
git clone git@github.com:Analytics4MD/A4MD-sample-workflow.git sampleEnsemble
cd sampleEnsemble
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DA4MD_PREFIX=$HOME/a4md-test -DDATASPACES_PREFIX=${app_install_dir}/dataspaces
make
make install
cd ../bin/
