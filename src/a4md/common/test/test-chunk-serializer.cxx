#include <catch2/catch.hpp>
#include <iostream>
#include <string>
#include "../include/chunk_serializer.h"

TEST_CASE( "ChunkSerializer Test on MDChunk", "[common]" ) 
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

    Chunk* chunk = new MDChunk(current_chunk_id, step, types, x_positions, y_positions, z_positions, lx, ly, lz, xy, xz, yz);

    std::string test_serialized_buffer;
    ChunkSerializer<SerializableChunk> chunk_serializer;

    test_serialized_buffer.clear();
    SerializableChunk serializable_chunk = SerializableChunk(chunk); 
    bool ret = chunk_serializer.serialize(serializable_chunk, test_serialized_buffer);
    
    REQUIRE(ret == true);
    REQUIRE(test_serialized_buffer.empty() == false);
    
    SerializableChunk de_serializable_chunk;
    ret = chunk_serializer.deserialize(de_serializable_chunk, test_serialized_buffer);
    REQUIRE(ret == true);
    
    Chunk* deserialized_chunk = de_serializable_chunk.get_chunk();
    MDChunk *deserialized_mdchunk = dynamic_cast<MDChunk *>(deserialized_chunk);
    // deserialized_mdchunk->print();
    MDChunk *mdchunk = dynamic_cast<MDChunk*>(chunk);
    // mdchunk->print();

    REQUIRE( chunk->get_chunk_id() == deserialized_chunk->get_chunk_id() );
    REQUIRE( mdchunk->get_timestep() == deserialized_mdchunk->get_timestep() );
    REQUIRE( mdchunk->get_types()[0] == deserialized_mdchunk->get_types()[0] );
    REQUIRE( mdchunk->get_x_positions()[0] == deserialized_mdchunk->get_x_positions()[0] );
    REQUIRE( mdchunk->get_box_lx() == deserialized_mdchunk->get_box_lx() );

    delete chunk;
    delete deserialized_chunk;
}

TEST_CASE( "ChunkSerializer Test on CVChunk", "[common]" ) 
{
    unsigned long int current_chunk_id = 0;
    std::vector<double> cv_values = { 0.1, 0.1, 0.1, 0.2, 0.2, 0.2 };

    Chunk* chunk = new CVChunk(current_chunk_id, cv_values);

    std::string test_serialized_buffer;
    ChunkSerializer<SerializableChunk> chunk_serializer;

    test_serialized_buffer.clear();
    SerializableChunk serializable_chunk = SerializableChunk(chunk); 
    bool ret = chunk_serializer.serialize(serializable_chunk, test_serialized_buffer);
    
    REQUIRE(ret == true);
    REQUIRE(test_serialized_buffer.empty() == false);
    
    SerializableChunk de_serializable_chunk;
    ret = chunk_serializer.deserialize(de_serializable_chunk, test_serialized_buffer);
    REQUIRE(ret == true);
    
    Chunk* deserialized_chunk = de_serializable_chunk.get_chunk();
    CVChunk *deserialized_cvchunk = dynamic_cast<CVChunk *>(deserialized_chunk);
    // deserialized_cvchunk->print();
    CVChunk *cvchunk = dynamic_cast<CVChunk*>(chunk);
    // cvchunk->print();

    REQUIRE( chunk->get_chunk_id() == deserialized_chunk->get_chunk_id() );
    REQUIRE( cvchunk->get_cv_values().size() == deserialized_cvchunk->get_cv_values().size() );
    REQUIRE( cvchunk->get_cv_values()[0] == deserialized_cvchunk->get_cv_values()[0] );

    delete chunk;
    delete deserialized_chunk;
}
