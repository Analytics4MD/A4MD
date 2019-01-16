#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <vector>
#include "py_runner.h"

TEST_CASE( "PyRunner ModuleLoadException Tests", "[retrieve]" )
{
    std::string m("dummy");
    std::string f("analyze");
    char* module_name = (char*)m.c_str();
    char* function_name = (char*)f.c_str();
    bool caught_py_exception = false;
    try
    {
      PyRunner runner = PyRunner(module_name,function_name);
    }
    catch(PythonModuleException ex)
    {
      caught_py_exception = true;
    }
    catch(...)
    {
    }

    REQUIRE( caught_py_exception == true );
}

TEST_CASE( "PyRunner Tests", "[retrieve]" )
{
    std::string m("test_analysis");
    std::string f("analyze");
    char* module_name = (char*)m.c_str();
    char* function_name = (char*)f.c_str();
    bool caught_py_exception = false;
    char cwd[256];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
      perror("getcwd() error");
    else
      printf("current working directory: %s \n", cwd);


    try
    {
      PyRunner runner = PyRunner(module_name,function_name);
      int types[3] = { 0, 0 ,0 };
      std::vector<double> x_positions = { 1.0, 2.0, 3.0 };
      double low, high;
      low = 0.0;
      high = 10.0;
      int step = 1;
      runner.analyze_frame(types,
                             x_positions,
                             x_positions,
                             x_positions,
                             low,
                             high,
                             low,
                             high,
                             low,
                             high,
                             step);
    }
    catch(PythonModuleException ex)
    {
      caught_py_exception = true;
    }
    catch(...)
    {
    }

    REQUIRE( caught_py_exception == false );
}
