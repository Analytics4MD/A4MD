#include "plumed_chunker.h"


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
    m_chunk_array.append(chunk);
}

ChunkArray PlumedChunker::get_chunk_array(int num_chunks)
{
    return m_chunk_array;
}

std::vector<Chunk> PlumedChunker::chunks_from_file(int num_chunks)
{
    throw NotImplementedException();
    std::vector<Chunk> temp;
    return temp;//m_chunks;
}
