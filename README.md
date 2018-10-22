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
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
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
3) Invoke the "run" function in "test_module" from a4md
```
./a4md/a4md -p test_module run 123
```
