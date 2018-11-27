#include "chunk_stager.h"


ChunkStager::ChunkStager(ChunkReader* reader, ChunkWriter* writer)
{
    this->m_chunk_reader = reader;
    this->m_chunk_writer = writer;
}

bool ChunkStager::stage_chunks(int num_chunks)
{
    bool success = false;
    std::vector<Chunk> chunks;
    throw NotImplementedException();
    return success; 
}
