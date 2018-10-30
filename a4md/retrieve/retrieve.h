#ifndef __RETRIEVE_H__
#define __RETRIEVE_H__
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

class Retrieve
{
    public:
        Retrieve();
        ~Retrieve();
        void run();
        int call_py(int, const char**);
};

#endif
