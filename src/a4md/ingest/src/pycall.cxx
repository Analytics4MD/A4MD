#include "../include/pycall.h"

PyObject *load_py_function(std::string py_script, std::string py_def){

  PyObject *py_retValue;
  PyObject *py_name, *py_module, *py_dict, *py_func = NULL;

  // Building name object
  py_name = PyUnicode_FromString(py_script.c_str()); 

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
  py_func = PyDict_GetItemString(py_dict, py_def.c_str());

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
