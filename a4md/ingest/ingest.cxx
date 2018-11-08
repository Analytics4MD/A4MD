#include "ingest.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_common.h>

PyObject *load_py_function(char *py_script, char *py_def){

  PyObject *py_retValue;
  PyObject *py_name, *py_module, *py_dict, *py_func = NULL;

  // Building name object
  py_name = PyUnicode_FromString(py_script); 

  // Loading module object
  py_module = PyImport_Import(py_name);
  if(py_module == NULL) {
    PyErr_Print();
    printf("py_module NULL\n");
    exit(-1);
  }

  // Get the python function/method name
  py_dict = PyModule_GetDict(py_module);
  if(py_dict == NULL) {
    PyErr_Print();
    printf("py_dict NULL\n");
    exit(-1);
  }

  // Sets python function to call below 
  py_func = PyDict_GetItemString(py_dict, py_def);

  if(py_func == NULL){
    PyErr_Print();
    printf("py_func NULL\n");
    exit(-1);
  }
  
  // Clean up Python references 
  Py_DECREF(py_module);
  Py_DECREF(py_name);
  Py_DECREF(py_dict);
  
  return py_func;
}

Ingest::Ingest()
{
    char* py_path = "./Python";
    char* py_script = "load";
    char* py_def = "extract_frame";
    //Initialize Python interpreter
    setenv("PYTHONPATH",py_path, 1);
    if (!Py_IsInitialized())
        Py_Initialize();

    m_py_func = load_py_function(py_script, py_def);
}

Ingest::~Ingest()
{
    // Python C extension finalization
    Py_Finalize();
}

void Ingest::run()
{
    printf("In Ingest Run\n");
}

int Ingest::extract_frame(char *file_name, char *log_name) {
    double *data = NULL;
    PyObject *py_retValue;
    PyObject *py_args;
    py_args = PyTuple_Pack(2, PyUnicode_FromString(file_name), PyUnicode_FromString(log_name));
    py_retValue = PyObject_CallObject(m_py_func, py_args);
    Py_DECREF(py_args);
    int nCA = 0;

    // Get partial CA coordinates 
    PyObject *py_num;
    py_num = PyList_GetItem(py_retValue, 0);
    if (PyLong_AsSsize_t(py_num) < 0) {
        Py_DECREF(py_retValue);
        return -1;
    } else {
        PyObject *py_CA;
        py_CA = PyList_GetItem(py_retValue, 1);
        int nCA;
        nCA = PyList_Size(py_CA);           
        data = (double *) malloc (nCA * 3 * sizeof (double));
        
        int i, j;
        PyObject *item;
        for (i = 0; i < nCA; i++) {
            item = PyList_GetItem(py_CA, i);
            for (j = 0; j < 3; j++) { 
                data[i * 3 + j] = PyFloat_AsDouble(PyTuple_GetItem(item, j));
            }
            Py_DECREF(item);
        }
        // Clean up
        Py_DECREF(py_CA);
        Py_DECREF(py_retValue);

        frame.data = data;
        frame.size = nCA;
        
        return 1;
    } 
}

