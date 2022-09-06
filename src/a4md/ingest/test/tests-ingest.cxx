#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <string>
#include <vector>
#include "../../common/include/chunk.h"
#include "../../common/include/md_runner.h"
#include "../include/pdb_chunker.h"
#include "../include/cv_buffer.h"

TEST_CASE("PyRunner extract_frame Tests", "[ingest]")
{
    std::string name = "ingest";
    std::string func = "extract_frame";
    // ToDo: unhardcode 
    std::string py_path = "./a4md/ingest/test";
    std::string file_path = "./a4md/ingest/test/test.pdb";

    int position = 0;
    int natoms = 0;
    MDRunner* py_runner = new MDRunner((char*)name.c_str(), 
                                       (char*)func.c_str(),
                                       (char*)py_path.c_str(), 
                                       (char*)file_path.c_str(), 
                                       natoms, 
                                       position);
    
    Chunk* chunk;
    unsigned long int id = 0;
    chunk = py_runner->output_chunk(id);
    position = py_runner->get_position();
    printf("New position : %d\n", position);

    MDChunk *plmdchunk = dynamic_cast<MDChunk *>(chunk);
    auto x_positions = plmdchunk->get_x_positions();
    auto y_positions = plmdchunk->get_y_positions();
    auto z_positions = plmdchunk->get_z_positions();
    auto types_vector = plmdchunk->get_types();
    int timestep = plmdchunk->get_timestep();

    // REQUIRE( result == 0 );
    REQUIRE( position == 4851 );
    REQUIRE( x_positions.size() == y_positions.size() );
    REQUIRE( y_positions.size() == z_positions.size() );
    REQUIRE( z_positions.size() == types_vector.size() );
    REQUIRE( timestep == 0);
}

TEST_CASE("PDBChunker Tests", "[ingest]")
{
    std::string name = "ingest";
    std::string func = "extract_frame";
    // ToDo: unhardcode 
    std::string py_path = "./a4md/ingest/test";
    std::string file_path = "./a4md/ingest/test/test.pdb";

    // PyRunner* py_runner = new MDRunner((char*)name.c_str(), 
    //                                    (char*)func.c_str(),
    //                                    (char*)py_path.c_str());
    
    PDBChunker* pdb_chunker = new PDBChunker((char*)name.c_str(), (char*)func.c_str(), (char*)py_path.c_str(), (char*)file_path.c_str(), 0 , 0);
    //unsigned long int id = 0;
    // int result = pdb_chunker->extract_chunk();
    std::vector<Chunk*> chunk_vector = pdb_chunker->read_chunks(1,1);
    Chunk* chunk = chunk_vector.front();
    MDChunk *plmdchunk = dynamic_cast<MDChunk *>(chunk);
    
    auto x_positions = plmdchunk->get_x_positions();
    auto y_positions = plmdchunk->get_y_positions();
    auto z_positions = plmdchunk->get_z_positions();
    auto types_vector = plmdchunk->get_types();
    int timestep = plmdchunk->get_timestep();

    // REQUIRE( result == 0 );
    //REQUIRE( pdb_chunker->get_position() == 4851 );
    REQUIRE( chunk_vector.size() == 1 );
    REQUIRE( x_positions.size() == y_positions.size() );
    REQUIRE( y_positions.size() == z_positions.size() );
    REQUIRE( z_positions.size() == types_vector.size() );
    REQUIRE( timestep == 0);
}

TEST_CASE("Knob nAtoms Tests", "[ingest]")
{
    std::string name = "ingest";
    std::string func = "extract_frame";
    // ToDo: unhardcode 
    std::string py_path = "./a4md/ingest/test";
    std::string file_path = "./a4md/ingest/test/test.pdb";

    // PyRunner* py_runner = new MDRunner((char*)name.c_str(), 
    //                                    (char*)func.c_str(),
    //                                    (char*)py_path.c_str());
    
    int natoms = 100;
    PDBChunker* pdb_chunker = new PDBChunker((char*)name.c_str(), (char*)func.c_str(), (char*)py_path.c_str(), (char*)file_path.c_str(), natoms, 0);
    // int result = pdb_chunker->extract_chunk();
    std::vector<Chunk*> chunk_vector = pdb_chunker->read_chunks(1,1);
    Chunk* chunk = chunk_vector.front();
    MDChunk *plmdchunk = dynamic_cast<MDChunk *>(chunk);
    //plmdchunk->print();

    auto x_positions = plmdchunk->get_x_positions();
    auto y_positions = plmdchunk->get_y_positions();
    auto z_positions = plmdchunk->get_z_positions();
    auto types_vector = plmdchunk->get_types();
    int timestep = plmdchunk->get_timestep();
    double box_lx = plmdchunk->get_box_lx();
    double box_ly = plmdchunk->get_box_ly();
    double box_lz = plmdchunk->get_box_lz();
    double box_hx = plmdchunk->get_box_hx();
    double box_hy = plmdchunk->get_box_hy();
    double box_hz = plmdchunk->get_box_hz();
    
    // REQUIRE( result == 0 );
    REQUIRE( chunk_vector.size() == 1 );
    REQUIRE( x_positions.size() == natoms );
    REQUIRE( x_positions.size() == y_positions.size() );
    REQUIRE( y_positions.size() == z_positions.size() );
    REQUIRE( z_positions.size() == types_vector.size() );
    REQUIRE( timestep == 0 );
    REQUIRE( box_lx == 0.0 );
    REQUIRE( box_ly == 0.0 );
    REQUIRE( box_lz == 0.0 );
    REQUIRE( box_hx == 0.0 );
    REQUIRE( box_hy == 0.0 );
    REQUIRE( box_hz == 0.0 );
}

TEST_CASE("CVBuffer Tests", "[ingest]")
{
    std::vector<double> cv_values = { 0.1, 0.1, 0.1, 0.2, 0.2, 0.2 };

    int interval = 2;
    ChunkOperator* cv_buffer = new CVBuffer(interval);
    
    for(unsigned long int chunk_id = 0; chunk_id < 10; chunk_id++)
    {
        Chunk* chunk = new CVChunk(chunk_id, cv_values);
        std::vector<Chunk*> input_chunks = {chunk};
        std::vector<Chunk*> output_chunks = cv_buffer->operate_chunks(input_chunks);
        if ((chunk_id + 1) % interval == 0)
        {
            REQUIRE( output_chunks.size() == 1 );
            Chunk* output_chunk = output_chunks.front();
            CVChunk *cv_chunk = dynamic_cast<CVChunk *>(output_chunks.front());
            // cv_chunk->print();
            REQUIRE( cv_chunk->get_cv_values().size() == 12);
            REQUIRE( cv_chunk->get_cv_values()[0] == 0.1);
            REQUIRE( cv_chunk->get_cv_values()[3] == 0.2);
            delete output_chunk;
        }
        if (chunk_id % interval == 0)
        {
            REQUIRE( output_chunks.size() == 0);
        }
        delete chunk;
    }
    delete cv_buffer;
}
