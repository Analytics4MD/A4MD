#include "chunk_stager.h"
#include <vector>

ChunkStager::ChunkStager(ChunkReader & chunk_reader, ChunkWriter & chunk_writer)
: m_chunk_reader(chunk_reader),
  m_chunk_writer(chunk_writer)
{
    printf("---===== Initalized ChunkStager with %s chunk_reader and %s chunk_writer\n", typeid(m_chunk_reader).name(), typeid(m_chunk_reader).name());
}

ChunkStager::~ChunkStager()
{
    printf("---===== Finalized ChunkStager\n");
}

bool ChunkStager::stage_chunks(int num_chunks)
{
    bool success = false;
    throw NotImplementedException("Need to refactor chunkstager to use from and to chunk if instead of num chunks.");
    //auto chunks = m_chunk_reader.read_chunks(num_chunks);
    //success = m_chunk_writer.write_chunks(chunks);
    //throw NotImplementedException();
    return success;
}

void ChunkStager::stage_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to)
{
    printf("ChunkStager::stage_chunks() --> Stage chunks from %lu to %lu\n", chunk_id_from, chunk_id_to);
    std::vector<Chunk*> chunks = m_chunk_reader.read_chunks(chunk_id_from, chunk_id_to);
    m_chunk_writer.write_chunks(chunks);
    for (Chunk* chunk : chunks)
    {
        free_chunk(chunk);
    }
}
