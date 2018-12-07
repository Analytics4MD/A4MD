#include "plumed_chunker.h"

std::vector<Chunk*> PlumedChunker::get_chunks(int num_chunks)
{
    if (m_chunkq.size() < num_chunks)
        throw "get_chunk_array asking for more chunks than what is available"; 
    std::vector<Chunk*> chunks;
    for (int i=0;i<num_chunks;i++)
    {
        chunks.push_back(m_chunkq.front());
        m_chunkq.pop();
    }
    return chunks;
}

void PlumedChunker::append(int step,
                           std::vector<int> types,
                           std::vector<double> x_cords,
                           std::vector<double> y_cords,
                           std::vector<double> z_cords,
                           double box_lx,
                           double box_ly,
                           double box_lz,
                           double box_xy,
                           double box_xz,
                           double box_yz)


{
    int id = step;//this needs to checked
    Chunk* chunk = new PLMDChunk(id,
                                 step,
                                 types,
                                 x_cords,
                                 y_cords,
                                 z_cords,
                                 box_lx,
                                 box_ly,
                                 box_lz,
                                 box_xy,
                                 box_xz,
                                 box_yz);
    m_chunkq.push(chunk);
}

//ChunkArray PlumedChunker::get_chunk_array(int num_chunks)
//{
//    if (m_chunkq.size() < num_chunks)
//        throw "get_chunk_array asking for more chunks than what is available"; 
//    ChunkArray chunk_ary;
//    for (int i=0;i<num_chunks;i++)
//    {
//        chunk_ary.append(m_chunkq.front());
//        m_chunkq.pop();
//    }
//    return chunk_ary;
//    //return m_chunk_array;
//}


