#include "chunk_stager.h"
#include <vector>
#ifdef TAU_PERF
#include <TAU.h>
#endif

ChunkStager::ChunkStager(ChunkReader* chunk_reader, std::vector<ChunkOperator*> chunk_operators, ChunkWriter* chunk_writer)
: m_chunk_reader(chunk_reader),
  m_chunk_writer(chunk_writer)
{
    m_chunk_operators = chunk_operators;
    printf("---===== Initalized ChunkStager\n");
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
    printf("---===== ChunkStager::stage_chunks() --> Stage chunks from %lu to %lu\n", chunk_id_from, chunk_id_to);
#ifdef TAU_PERF
    TAU_STATIC_TIMER_START("total_extract_chunks_time");
    TAU_DYNAMIC_TIMER_START("step_extract_chunks_time");
    //TAU_TRACK_MEMORY_FOOTPRINT();
    //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
#endif
    // Read chunks by ChunkReader
    std::vector<Chunk*> input_chunks = m_chunk_reader->read_chunks(chunk_id_from, chunk_id_to);

#ifdef TAU_PERF
    TAU_DYNAMIC_TIMER_STOP("step_extract_chunks_time");
    TAU_STATIC_TIMER_STOP("total_extract_chunks_time");

    TAU_STATIC_TIMER_START("total_write_chunks_time");
    TAU_DYNAMIC_TIMER_START("step_write_chunks_time");
#endif

    //Operate chunks by single/multiple ChunkOperators
    std::vector<Chunk*> output_chunks;
    for (ChunkOperator* chunk_operator : m_chunk_operators) {
        output_chunks = chunk_operator->operate_chunks(input_chunks);
        free_chunks(input_chunks);
        input_chunks = output_chunks;
    }

    // Write chunks by ChunkWriter
    m_chunk_writer->write_chunks(input_chunks);
    free_chunks(input_chunks);

#ifdef TAU_PERF
    TAU_DYNAMIC_TIMER_STOP("step_write_chunks_time");
    TAU_STATIC_TIMER_STOP("total_write_chunks_time");
#endif
}
