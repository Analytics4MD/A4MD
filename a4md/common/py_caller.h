#ifndef __PY_CALLER_H__
#define __PY_CALLER_H__
#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // pybind11 header needed for embedding
namespace py = pybind11;

class PyCaller 
{
	public:
		PyCaller();
		~PyCaller();
		py::object call(char* py_path, char* py_function);
};
#endif