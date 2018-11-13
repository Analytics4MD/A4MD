#ifndef __PYCALL_H__
#define __PYCALL_H__
#include <Python.h>
#include <string>

PyObject *load_py_function(std::string py_script, std::string py_def);

#endif