#include <catch2/catch.hpp>
#include <iostream>
#include <string>
#include "chunk.h"
// #include "chunk_container.h"
// #include "scalar_field_data.h"
#include "chunk_serializer.h"

std::string test_serialized_buffer;
Chunk* chunk;

// bool serialize_chunk_container();
// bool deserialize_chunk_container();
bool serialize_serializable_chunk();
bool deserialize_serializable_chunk();

TEST_CASE( "ChunkSerializer Test", "[common]" ) 
{

    // std::cout << "ChunkSerializer with ChunkContainer ..." << std::endl;
    // bool ret = serialize_chunk_container();
    // if (ret)
    //     deserialize_chunk_container();
    
    test_serialized_buffer.clear();
    printf("ChunkSerializer with SerializableChunk ...\n");
    bool ret = serialize_serializable_chunk();
    if (ret)
        deserialize_serializable_chunk();
}

// bool serialize_chunk_container()
// {
//     double test_value = 0.0;
//     std::shared_ptr<FieldData> test_data = std::make_shared<ScalarFieldData<double>>(test_value);
//     std::shared_ptr<FieldDataContainer> test_field_data_container = std::make_shared<FieldDataContainer>(test_data);

//     ChunkContainer chunk_container;
//     if (chunk_container.empty()) 
//     {
//         std::cout << "Chunk container is empty" << std::endl;
//     }
//     std::string test_field_name = "test_field";
//     chunk_container.add_field(test_field_name, test_field_data_container);
    
//     if (test_serialized_buffer.empty()) 
//     {
//         std::cout << "Buffer is empty at the beginning of serialization" << std::endl;
//     }
//     ChunkSerializer<ChunkContainer> chunk_serializer;
//     bool ret = chunk_serializer.serialize(chunk_container, test_serialized_buffer);
//     if (ret)
//     {
//         std::cout << "Sucessfully serialize test chunk container" << std::endl;
//         if (test_serialized_buffer.empty())
//         {
//             std::cout << "Buffer is still empty" << std::endl;
//         }
//     }
//     else 
//     {
//         std::cout << "Failed to serialize test chunk container" << std::endl;
//     }
//     return ret;
// }

// bool deserialize_chunk_container()
// {
//     double test_value = 5.0;
//     std::shared_ptr<FieldData> test_data = std::make_shared<ScalarFieldData<double>>(test_value);
//     std::shared_ptr<FieldDataContainer> test_field_data_container = std::make_shared<FieldDataContainer>(test_data);

//     ChunkContainer chunk_container;
//     std::string test_field_name = "test_field";
//     chunk_container.add_field(test_field_name, test_field_data_container);

//     if (chunk_container.empty())
//     {
//         std::cout << "New chunk container is empty at the beginning" << std::endl;
//     }
//     if (!test_serialized_buffer.empty()) 
//     {
//         std::cout << "Buffer is not empty at the beginning of deserialization" << std::endl;
//     }

//     ChunkSerializer<ChunkContainer> chunk_serializer;
//     bool ret = chunk_serializer.deserialize(chunk_container, test_serialized_buffer);
//     if (ret)
//     {
//         std::cout << "Sucessfully deserialize test chunk container" << std::endl;
//         if (!chunk_container.empty())
//         {
//             chunk_container.print();
//         }
//     }
//     else 
//     {
//         std::cout << "Failed to deserialize test chunk container" << std::endl;
//     }
//     return ret;
// }

bool serialize_serializable_chunk()
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

    chunk = new MDChunk(current_chunk_id,
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
    
    // MDChunk *md_chunk = dynamic_cast<MDChunk*>(chunk);
    // md_chunk->print();
    //chunk = &md_chunk;
    SerializableChunk serializable_chunk = SerializableChunk(chunk); 
    
    if (test_serialized_buffer.empty()) 
    {
        printf("Buffer is empty at the beginning of serialization\n");
    }

    ChunkSerializer<SerializableChunk> chunk_serializer;
    bool ret = chunk_serializer.serialize(serializable_chunk, test_serialized_buffer);
    if (ret)
    {
        printf("Sucessfully serialize test chunk container\n");
        if (test_serialized_buffer.empty())
        {
            printf("Buffer is still empty\n");
        }
    }
    else 
    {
        printf("Failed to serialize test chunk container\n");
    }
    return ret;
}

bool deserialize_serializable_chunk()
{
    if (!test_serialized_buffer.empty()) 
    {
        printf("Buffer is not empty at the beginning of deserialization\n");
    }

    SerializableChunk serializable_chunk;
    ChunkSerializer<SerializableChunk> chunk_serializer;
    bool ret = chunk_serializer.deserialize(serializable_chunk, test_serialized_buffer);
    if (ret)
    {
        printf("Sucessfully deserialize test chunk container\n");
        
        Chunk* deserialized_chunk = serializable_chunk.get_chunk();
        MDChunk *deserialized_mdchunk = dynamic_cast<MDChunk *>(deserialized_chunk);
        deserialized_mdchunk->print();
        MDChunk *mdchunk = dynamic_cast<MDChunk*>(chunk);
        mdchunk->print();

        REQUIRE( chunk->get_chunk_id() == deserialized_chunk->get_chunk_id() );
        REQUIRE( mdchunk->get_timestep() == deserialized_mdchunk->get_timestep() );
        REQUIRE( mdchunk->get_types()[0] == deserialized_mdchunk->get_types()[0] );
        REQUIRE( mdchunk->get_x_positions()[0] == deserialized_mdchunk->get_x_positions()[0] );
        REQUIRE( mdchunk->get_box_lx() == deserialized_mdchunk->get_box_lx() );
    }
    else 
    {
        printf("Failed to deserialize test chunk container\n");
    }
    return ret;
}