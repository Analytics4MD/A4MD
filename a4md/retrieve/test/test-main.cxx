#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <vector>
#include "md_runner.h"

TEST_CASE( "PyRunner ModuleLoadException Tests", "[retrieve]" )
{
    std::string m("dummy");
    std::string f("analyze");
    char* module_name = (char*)m.c_str();
    char* function_name = (char*)f.c_str();
    char* python_path = (char*)"";
    bool caught_py_exception = false;
    try
    {
        MDRunner runner = MDRunner(module_name,function_name,python_path);
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
    char* python_path = (char*)"./a4md/retrieve/test";
    bool caught_py_exception = false;
    char cwd[256];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
        perror("getcwd() error");
    else
        printf("current working directory: %s \n", cwd);


    try
    {
        MDRunner runner = MDRunner(module_name,function_name,python_path);
        std::vector<int> types = { 0, 0 ,0 };
        std::vector<double> x_positions = { 1.0, 2.0, 3.0 };
        double low, high;
        low = 0.0;
        high = 10.0;
        int step = 1;
        std::shared_ptr<Chunk> chunk = std::make_shared<MDChunk>(0,step,types,x_positions,x_positions,x_positions,low,low,low,high,high,high);
        runner.input_chunk(chunk);
        // delete chunk;
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
