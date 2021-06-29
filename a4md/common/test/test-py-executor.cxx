#include <catch2/catch.hpp>
#include <iostream>
#include <string>
#include "py_caller.h"
#include "chunks.h"

static PyCaller caller;

TEST_CASE( "PyCaller initilization Tests", "[common]" ) 
{
	REQUIRE(Py_IsInitialized() == 1);
}

TEST_CASE( "PyCaller call Tests", "[common]" ) 
{
	std::string py_path = "./a4md/common/test/tests.py";
	// PyCaller caller;
	py::object sum = caller.call((char*)py_path.c_str(), (char*)"add");
	REQUIRE(py::isinstance<py::function>(sum) == true);
	py::object result = sum(1,2);
	REQUIRE(py::isinstance<py::int_>(result) == true);
	REQUIRE(result.cast<int>() == 3);
	
	caller.call((char*)py_path.c_str(), (char*)"my_function");
}

TEST_CASE( "PyChunks module Tests", "[common]" )
{
	// PyCaller caller;
	std::string py_path = "./a4md/common/test/tests.py";
	std::string file_path = "./a4md/common/test/test.pdb";
	int position = 0;
    int num_atoms = 0;
	py::object func = caller.call((char*)py_path.c_str(), (char*)"extract_frame");
	REQUIRE(py::isinstance<py::function>(func) == true);
	py::tuple args = py::make_tuple(file_path, position, num_atoms);
	py::object result = func(*args);
	// REQUIRE(py::isinstance<py::class_<Chunk>>(result) == true);
	std::shared_ptr<Chunk> chunk = result.cast<std::shared_ptr<Chunk>>();
	chunk->print();
}

TEST_CASE( "Some tests", "[common]" ) 
{
	// PyCaller caller;
	std::string py_path = "./a4md/common/test/tests.py";
	caller.call((char*)py_path.c_str(), (char*)"my_function");
}
