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
    m_chunkq.push(chunk);
    //m_chunk_array.append(chunk);
}

ChunkArray PlumedChunker::get_chunk_array(int num_chunks)
{
    if (m_chunkq.size() < num_chunks)
        throw "get_chunk_array asking for more chunks than what is available"; 
    ChunkArray chunk_ary;
    for (int i=0;i<num_chunks;i++)
    {
        chunk_ary.append(m_chunkq.front());
        m_chunkq.pop();
    }
    return chunk_ary;
    //return m_chunk_array;
}

std::vector<Chunk> PlumedChunker::chunks_from_file(int num_chunks)
{
    throw NotImplementedException();
    std::vector<Chunk> temp;
    return temp;//m_chunks;
}
