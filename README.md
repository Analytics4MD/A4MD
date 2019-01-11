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
Now the executable a4md is in build/a4md.

This code detects the python and includes the header files. 
To demonstrate it, I have included a test function which can run a python function in a module. To test follow these steps:

1) Make a simple python file
```
# test_module.py
def run(arguments):
  print("Hello World", arguments)
```

2) Add the path to test_module.py to PYTHONPATH
```
# Typically done using export
export PYTHONPATH={PATH TO test_module.py}:$PYTHONPATH
```

3) Invoke the "run" function in "test_module" from a4md
```
./a4md/retriever -p test_module run 123
```

This should output "Hello world 123"
