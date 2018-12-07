#include "py_voronoi_analyzer.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_common.h>


PyVoronoiAnalyzer::PyVoronoiAnalyzer()
{
    initialize_python();
    printf("Initialized Retrieve\n");
}

PyVoronoiAnalyzer::~PyVoronoiAnalyzer()
{
    Py_DECREF(m_py_module);
    Py_DECREF(m_py_func); 
    Py_FinalizeEx(); 
}

int PyVoronoiAnalyzer::initialize_python()
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

  char* module_name = "calc_voronoi_for_frame";
  char* function_name = "analyze";
  m_py_module = PyImport_ImportModule(module_name);
  m_py_func = PyObject_GetAttrString(m_py_module, function_name);

  printf("-----===== Initialized python and the module ====-----\n");
  return 0;
}

 //! Used to analyze a frame of MD data using a python code.
 /*!
     \param module_name a python module name, e.g. "my_analysis_code".
     \param function_name a function name in the python module.
     
     \return integer value indicating success or failure  (0 is success, otherwise failure)
     \sa 
 */
int PyVoronoiAnalyzer::analyze_frame(char* module_name,
                            char* function_name,
                            int* types,
                            POS_VEC positions,
                            double x_low,
                            double x_high,
                            double y_low,
                            double y_high,
                            double z_low,
                            double z_high,
                            int step)
{
    int result = 0;
    if (!m_py_module)
    {
        PyErr_Print();
        fprintf(stderr,"import %s failed. See for an error message above\n",module_name);
        result = -1;
    }
    else
    {
        if (m_py_func && PyCallable_Check(m_py_func))
        {
            int count = positions.size();
            //for (int i=0;i<count;i++)
            //    printf("pos[%i]: %lf %lf %lf \n",std::get<0>(positions[i]),std::get<1>(positions[i]),std::get<2>(positions[i]));
            PyObject* py_args = PyTuple_New(4);
            npy_intp types_dims[] = {count};
            PyObject* py_types = PyArray_SimpleNewFromData(1, types_dims, NPY_DOUBLE, (void *)types);
            PyTuple_SetItem(py_args, 0, py_types);
 
            npy_intp positions_dims[] = {count, 3};
            PyObject* py_positions = PyArray_SimpleNewFromData(2, positions_dims, NPY_DOUBLE, static_cast<void*>(positions.data()));
            PyTuple_SetItem(py_args, 1, py_positions);

            //printf("C++ x_low %f %f %f %f %f %f\n",x_low,x_high,y_low,y_high,z_low,z_high);
            PyObject* py_box = Py_BuildValue("dddddd", x_low,x_high,y_low,y_high,z_low,z_high);
	    PyTuple_SetItem(py_args, 2, py_box);

            PyObject* py_step = Py_BuildValue("i", step);
	    PyTuple_SetItem(py_args, 3, py_step);

            PyObject* py_return = PyObject_CallObject(m_py_func, py_args);
            Py_DECREF(py_args);
            if (py_return != NULL)
            {
                //printf("Result of call: %ld\n", PyLong_AsLong(py_return));
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


