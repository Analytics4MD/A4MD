<h1 align="center">  
  A4MD
  <h4 align="center">
  <a href="https://app.shippable.com/github/Analytics4MD/A4MD-project-a4md"><img src="https://api.shippable.com/projects/5bcf364bec335d0700dbc0ec/badge?branch=master"/></a>
  </h4>
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#prerequisites">Prerequisites</a> •
  <a href="#dependencies">Dependencies</a> •
  <a href="#installation">Installation</a> 
</p>

## About

A framework that enables in situ molecular dynamic analytics via using in-memory staging areas

## Prerequisites

In order to use this package, your system should have the following installed:
- git
- cmake
- boost
- python

(Optional) To use the built-in analysis library, it is required to install:
- mdtraj
- freud

## Dependencies

The framework also builds the following external libraries as dependencies: 
- Dataspaces
- Catch2


## Installation

Here is the extensive installation instructions. Please make sure the all the prerequisites are satisfied before proceeding the following steps.

1. Clone the source code from this repository

```
git clone --recursive git@github.com:Analytics4MD/A4MD-project-a4md.git a4md
```

2. Build A4MD package 

Target build system should be specified in *-DTARGET_ARCH*. Please reference TargetArchList.txt for current support systems.

```
cd a4md
mkdir build
cd build

CC=$(which gcc) CXX=$(which g++) cmake .. \
-DCMAKE_INSTALL_PREFIX=${A4MD_ROOT} \
-DBOOST_ROOT=${BOOST_ROOT} \
-DPYTHON_EXECUTABLE=$(which python) \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))") \
-DMPI_C_COMPILER=$(which mpicc) \
-DMPI_CXX_COMPILER=$(which mpicxx) \
-DMPI_FORTRAN_COMPILER=$(which mpifort) \
-DTARGET_ARCH=${ARCH}

make
make install
```

*Note that* on Cori at NERSC, it is required to replace pure MPI compilers (mpicc, mpicxx, mpifort) with compiler wrappers (cc, CC, ftn), respectively.

3. Build A4MD Python package

```
cd a4md
pip install -e .
```

4. (Optional) To use TAU profiling in the code the cmake command can include the following flags. Of course, TAU needs to be installed on the system.

```
-DCMAKE_C_COMPILER=$(which tau_cc.sh) -DCMAKE_CXX_COMPILER=$(which tau_cxx.sh)
```
Please remember to unload darshan before installing TAU on Cori at NERSC as this package prevents TAU to work properly.

To use TAU manual instrumentation:

```
export TAU_TRACK_HEAP=1
export TAU_INTERRUPT_INTERVAL=1
export TAU_METRICS=TIME,PAPI_TOT_CYC,PAPI_TOT_INS,ENERGY
export TAU_LIBS=$(tau_cxx.sh -tau:showlibs)
export CXXFLAGS="-g -DPROFILING_ON -DTAU_STDCXXLIB -I${TAU_ROOT}/include"
```

5. (Optional) To use build-in performance scheme, run cmake command with the following flag:
```
-DBUILT_IN_PERF=ON
```

