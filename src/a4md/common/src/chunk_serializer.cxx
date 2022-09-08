#include "../include/chunk_serializer.h"

/* Boost headers */
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>


template<typename SerializableContainer>
ChunkSerializer<SerializableContainer>::ChunkSerializer()
{
    printf("---===== Initialized ChunkSerializer on %s\n", typeid(SerializableContainer).name());
}

template<typename SerializableContainer>
ChunkSerializer<SerializableContainer>::~ChunkSerializer()
{
    printf("---===== Finalized ChunkSerializer on %s\n", typeid(SerializableContainer).name());
}

template<typename SerializableContainer>
bool ChunkSerializer<SerializableContainer>::serialize(const SerializableContainer& serializable_container, std::string& serialized_buffer)
{
    if (!serialized_buffer.empty())
    {
        printf("---===== ERROR: ChunkSerializer::serialize() --> Serialized buffer is not empty\n");
        return false;
    }
    boost::iostreams::back_insert_device<std::string> inserter(serialized_buffer);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
    boost::archive::binary_oarchive oa(s);
    oa << serializable_container;
    s.flush();
    return true;
}

template<typename SerializableContainer>
bool ChunkSerializer<SerializableContainer>::deserialize(SerializableContainer& serializable_container, const std::string& serialized_buffer)
{
    if (serialized_buffer.empty())
    {
        printf("---===== ERROR: ChunkSerializer::deserialize() --> Serialized buffer is empty\n");
        return false;
    }
    boost::iostreams::basic_array_source<char> device(serialized_buffer.data(), serialized_buffer.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > sout(device);
    boost::archive::binary_iarchive ia(sout);
    ia >> serializable_container;
    return true;
}

template<typename SerializableContainer>
bool ChunkSerializer<SerializableContainer>::deserialize(SerializableContainer& serializable_container, const char* serialized_buffer, const size_t& buffer_size)
{
    if (serialized_buffer == nullptr)
    {
        printf("---===== ERROR: ChunkSerializer::deserialize() --> Serialized buffer is nullptr\n");
        return false;
    }
    boost::iostreams::basic_array_source<char> device(serialized_buffer, buffer_size);
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > sout(device);
    boost::archive::binary_iarchive ia(sout);
    ia >> serializable_container;
    return true;
}

template class ChunkSerializer<SerializableChunk>;

