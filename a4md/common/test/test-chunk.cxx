#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <vector>
#include "md_chunk.h"
#include "cv_chunk.h"
#include "md_runner.h"
#include "md_intermediator.h"

TEST_CASE( "MDChunk Tests", "[common]" ) 
{
    unsigned long int current_chunk_id = 0;
    int step = 1;
    std::vector<int> types = { 2, 1, 1 };
    std::vector<double> x_positions = { 0.1, 0.1, 0.1 };
    std::vector<double> y_positions = { 0.2, 0.2, 0.2 };
    std::vector<double> z_positions = { 0.2, 0.2, 0.2 };
    double lx, ly, lz, xy, xz, yz;
    lx=ly=lz=xy=xz=yz=1.0;
    lx=1.5;

    MDChunk md_chunk(current_chunk_id,
                           step,
                           types,
                           x_positions,
                           y_positions,
                           z_positions,
                           lx,
                           ly,
                           lz,
                           xy,
                           xz,
                           yz);
    Chunk* chunk = &md_chunk; 

    REQUIRE( chunk->get_chunk_id() == 0 );
    REQUIRE( md_chunk.get_timestep() == 1 );
    REQUIRE( md_chunk.get_types()[0] == 2 );
    REQUIRE( md_chunk.get_x_positions()[0] == 0.1 );
    REQUIRE( md_chunk.get_box_lx() == 1.5 );
}

TEST_CASE( "CVChunk Tests", "[common]" ) 
{
    unsigned long int current_chunk_id = 0;
    std::vector<double> cv_values = { 0.1, 0.1, 0.1, 0.2, 0.2, 0.2 };

    Chunk* chunk = new CVChunk(current_chunk_id, cv_values);
    CVChunk *cv_chunk = dynamic_cast<CVChunk*>(chunk);

    REQUIRE( chunk->get_chunk_id() == 0 );
    REQUIRE( cv_chunk->get_cv_values()[0] == 0.1 );
    REQUIRE( cv_chunk->get_cv_values()[3] == 0.2 );
    REQUIRE( cv_chunk->get_cv_values().size() == 6 );

    cv_chunk->append_cv_value(0.3);
    // cv_chunk->print();
    REQUIRE( cv_chunk->get_cv_values().size() == 7 );
    REQUIRE( cv_chunk->get_cv_values()[6] == 0.3 );

    delete chunk;
}

TEST_CASE( "MDIntermediator Tests", "[common]" )
{
    std::string m("test_direct");
    std::string f("direct");
    char* module_name = (char*)m.c_str();
    char* function_name = (char*)f.c_str();
    char* python_path = (char*)"./a4md/common/test";
    bool caught_py_exception = false;
    char cwd[256];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
        perror("getcwd() error");
    else
        printf("current working directory: %s \n", cwd);

    std::vector<int> input_types, output_types;
    std::vector<double> input_x_positions, output_x_positions;
    double input_low, input_high, output_low, output_high;
    int input_timestep, output_timestep;
    try
    {
        MDIntermediator *inter = new MDIntermediator(module_name,function_name,python_path);
        input_types = { 0, 0 ,0 };
        input_x_positions = { 1.0, 2.0, 3.0 };
        input_low = 0.0;
        input_high = 10.0;
        input_timestep = 1;
        Chunk *input_chunk = new MDChunk(0, 
            input_timestep, 
            input_types, 
            input_x_positions, 
            input_x_positions, 
            input_x_positions, 
            input_low, 
            input_low, 
            input_low, 
            input_high, 
            input_high, 
            input_high);
        std::vector<Chunk*> input_chunks = {input_chunk};
        std::vector<Chunk*> output_chunks = inter->operate_chunks(input_chunks);
        Chunk* output_chunk = output_chunks.front();
        MDChunk *plmdchunk = dynamic_cast<MDChunk *>(output_chunk);
        output_x_positions = plmdchunk->get_x_positions();
        output_types = plmdchunk->get_types();
        output_timestep = plmdchunk->get_timestep();
        output_low = plmdchunk->get_box_lx();
        output_high = plmdchunk->get_box_hx();

        delete input_chunk;
        delete output_chunk;
        delete inter;
    }
    catch(PythonModuleException ex)
    {
        caught_py_exception = true;
    }
    catch(...)
    {
    }

    REQUIRE( caught_py_exception == false );
    REQUIRE( output_x_positions.size() == input_x_positions.size() );
    REQUIRE( output_types.size() == input_types.size() );
    REQUIRE( output_timestep == input_timestep);
    REQUIRE( output_low == input_low);
    REQUIRE( output_high == input_high);
}