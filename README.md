[![Run Status](https://api.shippable.com/projects/5bcf364bec335d0700dbc0ec/badge?branch=master)]()
# A4MD

## Getting Started
```
git clone --recursive git@github.com:Analytics4MD/A4MD-project-a4md.git a4md
```

## Compiling
### Caliburn
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
To use tau profiling in the code the cmake command can include the following flags. Of course, tau needs to be installed on the system.

```
-DCMAKE_C_COMPILER=tau_cc.sh -DCMAKE_CXX_COMPILER=tau_cxx.sh
```
### Cori
Load module prerequisites
```
module swap PrgEnv-intel PrgEnv-gnu
module load python/3.6-anaconda-4.4
module load cmake/3.11.4
module load boost/1.69.0
```
Create Anaconda environement (i.e test_env), if not
```
source activate test_env
export LD_LIBRARY_PATH="$HOME/.conda/envs/test_env/lib:$LD_LIBRARY_PATH"
```
Build 
```
cd a4md
mkdir build
cd build
cmake .. \
-DCMAKE_INSTALL_PREFIX=../_install \
-DBOOST_ROOT=${BOOST_ROOT}
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))") \
-DMPI_C_COMPILER=$(which cc) \
-DMPI_CXX_COMPILER=$(which CC) \
-DMPI_FORTRAN_COMPILER=$(which ftn)

make
make install
```
Build A4MD Python package
```
cd a4md
pip install -e .
```
To use TAU manual instrumentation, install TAU at ${TAU_ROOT}
```
module unload darshan
module load papi
export TAU_TRACK_HEAP=1
export TAU_INTERRUPT_INTERVAL=1
export TAU_METRICS=TIME,PAPI_TOT_CYC,PAPI_TOT_INS,ENERGY
export TAU_LIBS=$(tau_cxx.sh -tau:showlibs)
export CXXFLAGS="-g -DPROFILING_ON -DTAU_STDCXXLIB -I${TAU_ROOT}/include"
```
To use build-in performance scheme, run cmake command with the following flag
```
-DBUILT_IN_PERF=ON
```
