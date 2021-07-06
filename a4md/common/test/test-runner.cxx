#include <catch2/catch.hpp>
#include "md_runner.h"
#include "chunks.h"
#include "cv_runner.h"
#include "py_caller.h"

TEST_CASE("MDRunner direct_chunk Tests", "[common]")
{
    std::string name = "direct";
    std::string func = "mdchunk_to_mdchunk";
    std::string py_path = "./a4md/common/test";

    PyRunner* py_runner = new MDRunner((char*)name.c_str(), 
                                       (char*)func.c_str(),
                                       (char*)py_path.c_str());
    bool caught_py_exception = false;
    try
    {
    	std::vector<int> input_types = { 0, 0 ,0 };
        std::vector<double> input_x_positions = { 1.0, 2.0, 3.0 };
        double input_low = 0.0;
        double input_high = 10.0;
        int input_timestep = 1;
        unsigned long int id = 0;
        std::shared_ptr<Chunk> input_chunk = std::make_shared<MDChunk>(id, input_timestep, input_types, input_x_positions, input_x_positions, input_x_positions, input_low, input_low, input_low, input_high, input_high, input_high);
    
    	std::shared_ptr<Chunk> output_chunk = py_runner->direct_chunk(input_chunk);
    	std::shared_ptr<MDChunk> plmdchunk = std::dynamic_pointer_cast<MDChunk>(output_chunk);

    	REQUIRE( plmdchunk->get_x_positions().size() == input_x_positions.size() );
	    REQUIRE( plmdchunk->get_types().size() == input_types.size() );
	    REQUIRE( plmdchunk->get_timestep() == input_timestep);
	    REQUIRE( plmdchunk->get_box_lx() == input_low);
	    REQUIRE( plmdchunk->get_box_hx() == input_high);

	    // delete input_chunk;
	    // delete output_chunk;
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

static PyCaller py_caller;
TEST_CASE("Direct md_chunk new Tests", "[common]")
{

    std::string py_function = "mdchunk_to_mdchunk";
    std::string py_script = "./a4md/common/test/tests.py";

    std::vector<int> input_types = { 0, 0 ,0 };
    std::vector<double> input_x_positions = { 1.0, 2.0, 3.0 };
    double input_low = 0.0;
    double input_high = 10.0;
    int input_timestep = 1;
    unsigned long int id = 0;

    std::shared_ptr<MDChunk> md_chunk = std::make_shared<MDChunk>(id, input_timestep, input_types, input_x_positions, input_x_positions, input_x_positions, input_low, input_low, input_low, input_high, input_high, input_high);
    std::shared_ptr<Chunk> input_chunk = md_chunk;
    py::object py_result = py_caller.call((char*)py_script.c_str(), (char*)py_function.c_str())(input_chunk);
    std::shared_ptr<Chunk> output_chunk = py_result.cast<std::shared_ptr<Chunk>>();
    output_chunk->print();
    std::shared_ptr<MDChunk> plmdchunk = std::dynamic_pointer_cast<MDChunk>(output_chunk);

    REQUIRE( plmdchunk->get_x_positions().size() == input_x_positions.size() );
    REQUIRE( plmdchunk->get_types().size() == input_types.size() );
    REQUIRE( plmdchunk->get_timestep() == input_timestep);
    REQUIRE( plmdchunk->get_box_lx() == input_low);
    REQUIRE( plmdchunk->get_box_hx() == input_high);
}

TEST_CASE("CVRunner direct_chunk Tests", "[common]")
{
    std::string name = "direct";
    std::string func = "mdchunk_to_cvchunk";
    std::string py_path = "./a4md/common/test";

    PyRunner* py_runner = new CVRunner((char*)name.c_str(), 
                                       (char*)func.c_str(),
                                       (char*)py_path.c_str());
    bool caught_py_exception = false;
    try
    {
    	std::vector<int> input_types = { 0, 0 ,0 };
        std::vector<double> input_x_positions = { 1.0, 2.0, 3.0 };
        double input_low = 0.0;
        double input_high = 10.0;
        int input_timestep = 1;
        unsigned long int id = 0;
        std::shared_ptr<Chunk> input_chunk = std::make_shared<MDChunk>(id, input_timestep, input_types, input_x_positions, input_x_positions, input_x_positions, input_low, input_low, input_low, input_high, input_high, input_high);
    
    	std::shared_ptr<Chunk> output_chunk = py_runner->direct_chunk(input_chunk);
    	std::shared_ptr<CVChunk> cv_chunk = std::dynamic_pointer_cast<CVChunk>(output_chunk);

    	REQUIRE( cv_chunk->get_cv_values().size() == input_x_positions.size() );
	    REQUIRE( cv_chunk->get_cv_values()[0] == input_x_positions[0] );

	    // delete input_chunk;
	    // delete output_chunk;
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

TEST_CASE("Direct cv_chunk new Tests", "[common]")
{

    std::string py_function = "mdchunk_to_cvchunk";
    std::string py_script = "./a4md/common/test/tests.py";

    std::vector<int> input_types = { 0, 0 ,0 };
    std::vector<double> input_x_positions = { 1.0, 2.0, 3.0 };
    double input_low = 0.0;
    double input_high = 10.0;
    int input_timestep = 1;
    unsigned long int id = 0;

    std::shared_ptr<MDChunk> md_chunk = std::make_shared<MDChunk>(id, input_timestep, input_types, input_x_positions, input_x_positions, input_x_positions, input_low, input_low, input_low, input_high, input_high, input_high);
    std::shared_ptr<Chunk> input_chunk = md_chunk;
    py::object py_result = py_caller.call((char*)py_script.c_str(), (char*)py_function.c_str())(input_chunk);
    std::shared_ptr<Chunk> output_chunk = py_result.cast<std::shared_ptr<Chunk>>();
    output_chunk->print();
    std::shared_ptr<CVChunk> cv_chunk = std::dynamic_pointer_cast<CVChunk>(output_chunk);

    REQUIRE( cv_chunk->get_cv_values().size() == input_x_positions.size() );
    REQUIRE( cv_chunk->get_cv_values()[0] == input_x_positions[0] );
}