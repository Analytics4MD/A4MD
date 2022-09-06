#include "../include/py_runner.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

// #define PY_ARRAY_UNIQUE_SYMBOL PY_ARRAY_API
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #include <numpy/arrayobject.h>
// #include <numpy/ndarrayobject.h>
// #include <numpy/npy_common.h>


PyRunner::PyRunner(char* module_name, char* function_name, char* py_path)
: m_module_name(module_name),
  m_function_name(function_name)
{
    initialize_python(py_path);
    printf("---===== Initialized PyRunner\n");
}

PyRunner::~PyRunner()
{
    Py_DECREF(m_py_module);
    Py_DECREF(m_py_func);
    /* Note: Python cannot completely unload extension modules, which prevent the interpreter from restarting when calling Py_Initialize second time */ 
    Py_FinalizeEx(); 
    printf("---===== Finalized PyRunner\n");
}

void PyRunner::print_py_error_and_rethrow()
{
    if(PyErr_Occurred())
    {
        std::string err = "\n";
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        if (type != NULL)
        {
            PyObject* str_exc_type = PyObject_Repr(type); //Now a unicode object
            PyObject* pyStr = PyUnicode_AsEncodedString(str_exc_type, "utf-8","Error ~");
            const char *strExcType =  PyBytes_AS_STRING(pyStr);
            printf("%s\n",strExcType);
            std::string err_type(strExcType);
            err = "Exception type: "+err_type+"\n";
        }
        if (value != NULL)
        {
            PyObject* str_exc_type = PyObject_Repr(value); //Now a unicode object
            PyObject* pyStr = PyUnicode_AsEncodedString(str_exc_type, "utf-8","Error ~");
            const char *strExcVal =  PyBytes_AS_STRING(pyStr);
            printf("%s\n",strExcVal);
            std::string err_val(strExcVal);
            err = err + "Exception value: "+ err_val+"\n";
        }
        if (traceback != NULL)
        {
            PyObject* str_exc_type = PyObject_Repr(traceback); //Now a unicode object
            PyObject* pyStr = PyUnicode_AsEncodedString(str_exc_type, "utf-8","Error ~");
            const char *strExcTraceBack =  PyBytes_AS_STRING(pyStr);
            printf("%s\n",strExcTraceBack);
            std::string err_trace(strExcTraceBack);
            err = err + "Exception traceback: "+ err_trace+"\n";
        }
        throw PythonModuleException(err.c_str());
    }
}

int PyRunner::initialize_python(char* py_path)
{
  //Initialize Python interpreter
    if (!Py_IsInitialized())
    {
        Py_Initialize();
        // Py_InitializeEx(1);
    }

    // import_array();
    // PyObject* module = PyImport_ImportModule("numpy"); // New reference
    // if (!module)
    // {
    //     PyErr_Print();
    //     fprintf(stderr,"---===== ERROR : PyRunner::initialize_python Numpy import failed. See for an error message above");
    // } 
    // Py_DECREF(module);

    char cwd[256];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
    {
        perror("getcwd() error");
    }
    else
    {
        printf("current working directory: %s \n", cwd);
    }

    PyObject* sysPath = PySys_GetObject((char*)"path");
    PyList_Append(sysPath, PyUnicode_FromString("."));
    PyList_Append(sysPath, PyUnicode_FromString(py_path));
    printf("---===== User defined python path: %s\n", py_path);
    printf("---===== Import module : %s\n",m_module_name);
    m_py_module = PyImport_ImportModule(m_module_name);
    if (m_py_module ==NULL)
    {
        printf("---===== ERROR: PyRunner::initialize_python Import module %s failed\n",m_module_name);
        print_py_error_and_rethrow();
    }

    printf("---===== Import module %s done\n",m_module_name);
    m_py_func = PyObject_GetAttrString(m_py_module, m_function_name);

    if (m_py_func && PyCallable_Check(m_py_func))
    {
        printf("---===== Initialized python and the module\n");
    }
    else
    {
        printf("---===== ERROR : PyRunner::initialize_python Could not load %s in %s. Please check if the function signature matches specification\n",m_module_name,m_function_name); 
        print_py_error_and_rethrow();
    }
    return 0;
}

