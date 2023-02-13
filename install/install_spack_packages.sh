#!/usr/bin/env bash

# echo "Downloading and installing Spack"
# git clone -c feature.manyFiles=true https://github.com/spack/spack.git ~/spack
# cd ~/spack
# git checkout releases/v0.18
# . share/spack/setup-env.sh

echo "Creating spack environment"
spack env create a4md_spack_env
spack env activate a4md_spack_env

echo 'Installing modules with spack' 
spack install gcc@5.5.0
spack load gcc@5.5.0
spack compiler add $(spack location -i gcc@5.5.0)

spack install mpich %gcc@5.5.0
spack load mpich %gcc@5.5.0

spack install boost %gcc@5.5.0 cxxstd=11 +iostreams +serialization
spack load boost %gcc@5.5.0 cxxstd=11 +iostreams +serialization

spack install cmake %gcc@5.5.0
spack load cmake %gcc@5.5.0

$(spack location -i pip3) install numpy

export BOOST_ROOT=$(spack location -i boost)
export LD_LIBRARY_PATH="${BOOST_ROOT}/lib:${LD_LIBRARY_PATH}"

