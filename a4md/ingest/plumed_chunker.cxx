#include "plumed_chunker.h"
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/serialization/vector.hpp>


PlumedChunker::PlumedChunker()
{
}

PlumedChunker::~PlumedChunker() 
{
}

void PlumedChunker::initialize()
{
}

void PlumedChunker::finalize()
{
}

void PlumedChunker::append(int step,
                           std::vector<int> types,
                           std::vector<double> x_cords,
                           std::vector<double> y_cords,
                           std::vector<double> z_cords)
{
    PLMDChunk* chunk = new PLMDChunk(step, types);
    m_chunks.insert(m_chunks.end(), chunk);    
    m_chunk_array.append(chunk);
}

ChunkArray PlumedChunker::get_chunk_array(int num_chunks)
{
    return m_chunk_array;
}
std::vector<Chunk> PlumedChunker::chunks_from_file(int num_chunks)
{
    //if (m_chunks.size() > num_chunks)
    std::vector<Chunk> temp;
    return temp;//m_chunks;
    //TODO else throw exception
}
