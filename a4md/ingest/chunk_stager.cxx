#include "chunk_stager.h"
#include <vector>
#ifdef TAU_PERF
#include <TAU.h>
#endif

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
#ifdef TAU_PERF
    TAU_STATIC_TIMER_START("total_extract_chunks_time");
    TAU_DYNAMIC_TIMER_START("step_extract_chunks_time");
#endif
    std::vector<Chunk*> chunks = m_chunk_reader.read_chunks(chunk_id_from, chunk_id_to);
#ifdef TAU_PERF
    TAU_DYNAMIC_TIMER_STOP("step_extract_chunks_time");
    TAU_STATIC_TIMER_STOP("total_extract_chunks_time");

    TAU_STATIC_TIMER_START("total_write_chunks_time");
    TAU_DYNAMIC_TIMER_START("step_write_chunks_time");
#endif
    m_chunk_writer.write_chunks(chunks);
    for (Chunk* chunk : chunks)
    {
        free_chunk(chunk);
    }
#ifdef TAU_PERF
    TAU_DYNAMIC_TIMER_STOP("step_write_chunks_time");
    TAU_STATIC_TIMER_STOP("total_write_chunks_time");
#endif
}
