#include "py_caller.h"
#include <string>

PyCaller::PyCaller()
{
	if (!Py_IsInitialized())
    {
		py::initialize_interpreter();
	}
	printf("(%s): initialized\n",__func__);
	
}

PyCaller::~PyCaller()
{
	
	py::finalize_interpreter();
	printf("(%s): finalized\n", __func__);
}

py::object PyCaller::call(char* py_script, char* py_function)
{
	// py::scoped_interpreter guard{};
	std::string py_script_str(py_script);
	std::size_t module_start = py_script_str.find_last_of("/");
	std::size_t module_end = py_script_str.rfind(".py");
	if (module_end == std::string::npos)
	{
		fprintf(stderr, "ERROR: Expecting a python module in the py_script argument.\n");
	    exit(1);
	}
	// get directory
	std::string py_dir = (std::string::npos==module_start)? std::string(".") : py_script_str.substr(0,module_start);
	// get file
	std::string py_name = py_script_str.substr(module_start+1, module_end-module_start-1);
	printf("(%s): Python directory : %s\n", __func__, py_dir.c_str());
    printf("(%s): Python script name : %s\n", __func__, py_name.c_str());
    printf("(%s): Python function: %s\n", __func__, py_function);
	py::list sys_path = py::module_::import("sys").attr("path");
	sys_path.append(py_dir);
	return py::module_::import((char*)py_name.c_str()).attr(py_function);
}