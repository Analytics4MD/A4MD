#include "retrieve.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_common.h>


Retrieve::Retrieve()
{
    initialize_python();
    printf("Initialized Retrieve\n");
}

Retrieve::~Retrieve()
{
    Py_FinalizeEx(); 
}

int Retrieve::initialize_python()
{
  //Initialize Python interpreter
  if (!Py_IsInitialized())
    Py_Initialize();
 
  
  import_array();
  PyObject* module = PyImport_ImportModule("numpy"); // New reference
  if (!module)
  {
    PyErr_Print();
    fprintf(stderr,"numpy import failed. See for an error message above");
  } 
  Py_DECREF(module);

  char cwd[256];
  if (getcwd(cwd, sizeof(cwd)) == NULL)
    perror("getcwd() error");
  else
    printf("Python modules in current working directory: %s will be found automatically,\
    but not python modules in other locations. Please add those paths to PYTHONPATH.\n", cwd);

  PyObject* sysPath = PySys_GetObject((char*)"path");
  PyList_Append(sysPath, PyUnicode_FromString("."));

  return 0;
}

void Retrieve::run()
{
    printf("In Retrieve Run\n");
}

int Retrieve::aanalyze_frame(std::vector<std::tuple<double, double, double> > module_name){return 0;}

 //! Used to analyze a frame of MD data using a python code.
 /*!
     \param module_name a python module name, e.g. "my_analysis_code".
     \param function_name a function name in the python module.
     
     \return integer value indicating success or failure  (0 is success, otherwise failure)
     \sa 
 */
int Retrieve::analyze_frame(char* module_name,
                            char* function_name,
                            int* types,
                            POS_VEC positions,
                            double x_low,
                            double x_high,
                            double y_low,
                            double y_high,
                            double z_low,
                            double z_high)
{
    int result = 0;
    PyObject* py_module = PyImport_ImportModule(module_name);
    if (!py_module)
    {
        PyErr_Print();
        fprintf(stderr,"import %s failed. See for an error message above\n",module_name);
        result = -1;
    }
    else
    {
        Py_DECREF(py_module);
        PyObject* py_func = PyObject_GetAttrString(py_module, function_name);
        if (py_func && PyCallable_Check(py_func))
        {
            int count = positions.size();
            Py_DECREF(py_func); 
            PyObject* py_args = PyTuple_New(3);
            npy_intp types_dims[] = {count};
            PyObject* py_types = PyArray_SimpleNewFromData(1, types_dims, NPY_DOUBLE, (void *)types);
            PyTuple_SetItem(py_args, 0, py_types);
 
            npy_intp positions_dims[] = {count, 3};
            PyObject* py_positions = PyArray_SimpleNewFromData(2, positions_dims, NPY_DOUBLE, static_cast<void*>(positions.data()));
            PyTuple_SetItem(py_args, 1, py_positions);

            //printf("C++ x_low %f %f %f %f %f %f\n",x_low,x_high,y_low,y_high,z_low,z_high);
            PyObject* py_box = Py_BuildValue("dddddd", x_low,x_high,y_low,y_high,z_low,z_high);
	    PyTuple_SetItem(py_args, 2, py_box);
            PyObject* py_return = PyObject_CallObject(py_func, py_args);
            Py_DECREF(py_args);
            if (py_return != NULL)
            {
                printf("Result of call: %ld\n", PyLong_AsLong(py_return));
                Py_DECREF(py_return);
            }
            else
            {
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                result = 1;
            }


        }
        else
        {
            fprintf(stderr,"Python function %s is not found in %s\n",function_name, module_name);
            result = -2;
        }

    } 
    return result;
}

int Retrieve::call_py(int argc, const char** argv)
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  if (argc < 3) {
      fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
      return 1;
  }
  
  printf("Retrieve::call_py called %s, %s, %s\n",argv[0],argv[1],argv[2]); 
 
  pName = PyUnicode_DecodeFSDefault(argv[1]);
  /* Error checking of pName left out */
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL) {
      pFunc = PyObject_GetAttrString(pModule, argv[2]);
      /* pFunc is a new reference */

      if (pFunc==NULL){
          printf("pFunc is NULL. func name is %s\n",argv[2]);
      }
      if (pFunc && PyCallable_Check(pFunc)) {
          pArgs = PyTuple_New(argc+1 - 3);
          for (i = 0; i < argc - 3; ++i) {
              pValue = PyLong_FromLong(atoi(argv[i + 3]));
              if (!pValue) {
                  Py_DECREF(pArgs);
                  Py_DECREF(pModule);
                  fprintf(stderr, "Cannot convert argument\n");
                  return 1;
              }
              /* pValue reference stolen here: */
              PyTuple_SetItem(pArgs, i, pValue);
          }
          const int nrow = 3, ncol = 4, nelem = nrow*ncol;
          double** m = new double*[nrow];
          PyObject *mat;
          m[0] = new double[nelem];
          m[1] = m[0] + ncol;
          m[2] = m[1] + ncol;				
          
          // fill in values				
          m[0][0] = 1.0; m[0][1] = 2.0; m[0][2] = 5.0; m[0][3] = 34;
          m[1][0] = 5.0; m[1][1] = 1.0; m[1][2] = 8.0; m[1][3] = 64;
          m[2][0] = 8.0; m[2][1] = 0.0; m[2][2] = 3.0; m[2][3] = 12;
          npy_intp mdim[] = { nrow, ncol };
	  printf("Going to call PyAttay_SimpleNewFromData\n");
          
          POS_VEC positions;
          positions.push_back(std::make_tuple(1.0,1.0,1.0));
          positions.push_back(std::make_tuple(1.0,1.0,1.0));
          std::cout<< "positions size" << positions.size()<<"\n";
          int num_atoms = positions.size();
          npy_intp positions_dims[] = { num_atoms, 3 };
          PyObject* py_positions = PyArray_SimpleNewFromData(2, positions_dims, NPY_DOUBLE, static_cast<void*>(positions.data()));
          PyTuple_SetItem(pArgs, 1, py_positions);

          //mat = PyArray_SimpleNewFromData(2, mdim, NPY_DOUBLE, (void *)m[0]);
          //PyTuple_SetItem(pArgs, 1, mat);

          pValue = PyObject_CallObject(pFunc, pArgs);
          Py_DECREF(pArgs);
          if (pValue != NULL) {
              printf("Result of call: %ld\n", PyLong_AsLong(pValue));
              Py_DECREF(pValue);
          }
          else {
              Py_DECREF(pFunc);
              Py_DECREF(pModule);
              PyErr_Print();
              fprintf(stderr,"Call failed\n");
              return 1;
          }
      }
      else {
          if (PyErr_Occurred())
              PyErr_Print();
          fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
      }
      Py_XDECREF(pFunc);
      Py_DECREF(pModule);
  }
  else {
      PyErr_Print();
      fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
      return 1;
  }
  return 0;
}
