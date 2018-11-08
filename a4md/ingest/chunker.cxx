#include "chunker.h"

PdbChunker::PdbChunker(std::string file_path) 
{
    m_file_path = file_path;
}

PdbChunker::~PdbChunker()
{
}

std::vector<Chunk> PdbChunker::chunks_from_file(std::string file_path, int num_chunks)
{
    std::vector<Chunk> chunks;
    throw NotImplementedException();
    //read from persistent storage
    //logic to iterate from last read chunk to num_chunks
    //ch1.data = python_get_frame();
    return chunks;
}
