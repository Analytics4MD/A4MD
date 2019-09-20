<h1 align="center">  
  A4MD
  <h4 align="center">
  <a href="https://app.shippable.com/github/Analytics4MD/A4MD-project-a4md"><img src="https://api.shippable.com/projects/5bcf364bec335d0700dbc0ec/badge?branch=master"/></a>
  </h4>
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#dependencies">Dependencies</a>
</p>

## About
A framework that enables in situ molecular dynamic analytics via using in-memory staging areas

## Dependencies
- [Boost](https://www.boost.org)
- [mdtraj](http://mdtraj.org)
- [Dataspaces](http://www.dataspaces.org)
- [Catch2](https://github.com/catchorg/Catch2)
- (Optional) [Plumed2](https://github.com/plumed/plumed2)

## Installation

Here is the extensive installation instructions on several HPC computer systems.

### Getting Started

<details><summary><b>Show instructions</b></summary>
  
Clone the source code from this repository

```
git clone --recursive git@github.com:Analytics4MD/A4MD-project-a4md.git a4md
```

</details>

### Caliburn

<details><summary><b>Show instructions</b></summary>

1. Build A4MD package 
```
cd a4md
mkdir build
cd build

module purge
module load python/3.6.3
module load openmpi/2.1.3-gcc-4.8.5
module load boost/1.68-gcc-4.8.5
cmake .. \
-DCMAKE_INSTALL_PREFIX=../_install \
-DBOOST_ROOT=/software/boost/1.68-gcc-4.8.5 \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))")
```
2. (Optional) To use tau profiling in the code the cmake command can include the following flags. Of course, tau needs to be installed on the system.

```
-DCMAKE_C_COMPILER=tau_cc.sh -DCMAKE_CXX_COMPILER=tau_cxx.sh
```
</details>

### Cori

<details><summary><b>Show instructions</b></summary>

1. Load module prerequisites

Note: boost/1.70.0 is currently not able to be found by find_package in cmake. It is recommended to use boost/1.69.0 until the issue is resolved.
```
module swap PrgEnv-intel PrgEnv-gnu
module load python/3.7-anaconda-2019.07
module load cmake
module load boost/1.69.0
module load rdma-credentials
```
2. Create Anaconda environement (i.e test_env), if not.
```
conda create -n ${A4MD_ENV}
```
3. Load created Python environment 
```
source activate ${A4MD_ENV}
export LD_LIBRARY_PATH="$HOME/.conda/envs/${A4MD_ENV}/lib:$LD_LIBRARY_PATH"
```
4. Install Python dependencies
```
conda install -c conda-forge mdtraj
```
5. Build A4MD package 
```
cd a4md
mkdir build
cd build
cmake .. \
-DCMAKE_INSTALL_PREFIX=../_install \
-DBOOST_ROOT=${BOOST_ROOT} \
-DPYTHON_EXECUTABLE=$(which python) \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))") \
-DMPI_C_COMPILER=$(which cc) \
-DMPI_CXX_COMPILER=$(which CC) \
-DMPI_FORTRAN_COMPILER=$(which ftn)

make
make install
```
6. Build A4MD Python package
```
cd a4md
pip install -e .
```
7. (Optional) To use TAU manual instrumentation, install TAU at ${TAU_ROOT}
```
module unload darshan
module load papi
export TAU_TRACK_HEAP=1
export TAU_INTERRUPT_INTERVAL=1
export TAU_METRICS=TIME,PAPI_TOT_CYC,PAPI_TOT_INS,ENERGY
export TAU_LIBS=$(tau_cxx.sh -tau:showlibs)
export CXXFLAGS="-g -DPROFILING_ON -DTAU_STDCXXLIB -I${TAU_ROOT}/include"
```
8. (Optional) To use build-in performance scheme, run cmake command with the following flag
```
-DBUILT_IN_PERF=ON
```
</details>

