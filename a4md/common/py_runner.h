#ifndef __PY_RUNNER_H__
#define __PY_RUNNER_H__
#include <vector>
#define PY_SSIZE_T_CLEAN
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

        virtual void input_chunk(std::shared_ptr<Chunk> chunk) = 0;
        virtual std::shared_ptr<Chunk> output_chunk(unsigned long int chunk_id) = 0;
        virtual std::shared_ptr<Chunk> direct_chunk(std::shared_ptr<Chunk> chunk) = 0;
        
};
#endif
