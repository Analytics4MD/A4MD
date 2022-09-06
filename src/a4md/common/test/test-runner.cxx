#include <catch2/catch.hpp>
#include "../include/md_runner.h"
#include "../include/cv_chunk.h"
#include "../include/cv_runner.h"

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
        Chunk *input_chunk = new MDChunk(id, input_timestep, input_types, input_x_positions, input_x_positions, input_x_positions, input_low, input_low, input_low, input_high, input_high, input_high);
    
    	Chunk* output_chunk = py_runner->direct_chunk(input_chunk);
    	MDChunk *plmdchunk = dynamic_cast<MDChunk *>(output_chunk);

    	REQUIRE( plmdchunk->get_x_positions().size() == input_x_positions.size() );
	    REQUIRE( plmdchunk->get_types().size() == input_types.size() );
	    REQUIRE( plmdchunk->get_timestep() == input_timestep);
	    REQUIRE( plmdchunk->get_box_lx() == input_low);
	    REQUIRE( plmdchunk->get_box_hx() == input_high);

	    delete input_chunk;
	    delete output_chunk;
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
        Chunk *input_chunk = new MDChunk(id, input_timestep, input_types, input_x_positions, input_x_positions, input_x_positions, input_low, input_low, input_low, input_high, input_high, input_high);
    
    	Chunk* output_chunk = py_runner->direct_chunk(input_chunk);
    	CVChunk *cv_chunk = dynamic_cast<CVChunk *>(output_chunk);

    	REQUIRE( cv_chunk->get_cv_values().size() == input_x_positions.size() );
	    REQUIRE( cv_chunk->get_cv_values()[0] == input_x_positions[0] );

	    delete input_chunk;
	    delete output_chunk;
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
