#ifndef __RETRIEVER_H__
#define __RETRIEVER_H__
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

class Retriever
{
    public:
        Retriever();
        ~Retriever();
        void run();
        int call_py(int, const char**);
};

#endif
