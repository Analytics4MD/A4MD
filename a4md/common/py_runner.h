#ifndef __PY_RUNNER_H__
#define __PY_RUNNER_H__
#include <vector>
#include <Python.h>
#include "exceptions.h"
#include "md_chunk.h"

class PyRunner
{
    protected:
        PyObject* m_py_module;
        PyObject* m_py_func;
        const char* m_module_name;
        const char* m_function_name;

        int initialize_python(char* py_path);
    public:
        PyRunner(char* module_name,
				 char* function_name,
				 char* py_path = (char*)"");
        ~PyRunner();

        void print_py_error_and_rethrow();

        virtual void input_chunk(Chunk* chunk) = 0;
        virtual Chunk* output_chunk(int chunk_id) = 0;
        
        // virtual int analyze_frame(std::vector<int> types,
        //                   std::vector<double> x_positions,
        //                   std::vector<double> y_positions,
        //                   std::vector<double> z_positions,
        //                   double x_low,
        //                   double x_high,
        //                   double y_low,
        //                   double y_high,
        //                   double z_low,
        //                   double z_high,
        //                   int step) = 0;
        // virtual int extract_frame(char* file_path,
        //                   unsigned long int id,
        //                   int &position,
        //                   Chunk **chunk,
        //                   int natoms = 0) = 0;
};
#endif
