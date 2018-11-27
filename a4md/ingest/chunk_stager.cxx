#include "chunk_stager.h"


ChunkStager::ChunkStager(ChunkReader* reader, ChunkWriter* writer)
{
    m_chunk_reader = reader;
    m_chunk_writer = writer;
}

bool ChunkStager::stage_chunks(int num_chunks)
{
    bool success = false;
    auto chunks = m_chunk_reader->read_chunks(num_chunks);
    success = m_chunk_writer->write_chunks(chunks);
    //throw NotImplementedException();
    return success; 
}
