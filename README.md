[![Run Status](https://api.shippable.com/projects/5bcf364bec335d0700dbc0ec/badge?branch=master)]()
# A4MD

## Getting Started
```
git clone --recursive git@github.com:Analytics4MD/A4MD-project-a4md.git a4md
```

## Compiling
```
cd a4md
mkdir build
cd build

----------==============  Caliburn cluster ==============--------------
module purge
module load python/3.6.3
module load openmpi/2.1.3-gcc-4.8.5
module load boost/1.68-gcc-4.8.5
cmake .. \
-DCMAKE_INSTALL_PREFIX=../_install \
-DBOOST_ROOT=/software/boost/1.68-gcc-4.8.5 \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))")
----------==============  Caliburn cluster ==============--------------
make
```
To use tau profiling in the code the cmake command can include the following flags. Of course, tau needs to be installed on the system.

```
-DCMAKE_C_COMPILER=tau_cc.sh -DCMAKE_CXX_COMPILER=tau_cxx.sh
```
