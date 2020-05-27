#ifndef __PY_RUNNER_H__
#define __PY_RUNNER_H__
#include <vector>
#include <Python.h>
#include "exceptions.h"
#include "md_chunk.h"

class PyRunner
{
    private:
        int initialize_python(char* py_path);
        PyObject* m_py_module;
        PyObject* m_py_func;
        const char* m_module_name;
        const char* m_function_name;
    public:
        PyRunner(char* module_name,
				 char* function_name,
				 char* py_path = (char*)"");
        ~PyRunner();
        
        int analyze_frame(std::vector<int> types,
                          std::vector<double> x_positions,
                          std::vector<double> y_positions,
                          std::vector<double> z_positions,
                          double x_low,
                          double x_high,
                          double y_low,
                          double y_high,
                          double z_low,
                          double z_high,
                          int step);
        int extract_frame(char* file_path,
                          unsigned long int id,
                          int &position,
                          Chunk **chunk,
                          int natoms = 0);
};
#endif
