#include "../include/chunk_analyzer.h"
#include <typeinfo>
#ifdef TAU_PERF
#include <TAU.h>
#endif

ChunkAnalyzer::ChunkAnalyzer(ChunkReader & chunk_reader)
: m_chunk_reader(chunk_reader)
{
    printf("---===== Initialized ChunkAnalyzer with %s chunk_reader\n", typeid(m_chunk_reader).name());
}

ChunkAnalyzer::~ChunkAnalyzer()
{
    printf("---===== Finalized ChunkAnalyzer\n");
}
void ChunkAnalyzer::analyze_chunks(int chunk_id_from, int chunk_id_to)
{
    printf("---===== ChunkAnalyzer::analyze_chunks() --> Analyze chunks from chunk_id_from = %lu to chunk_id_to = %lu\n");
#ifdef TAU_PERF
    TAU_STATIC_TIMER_START("total_read_chunks_time");
    TAU_DYNAMIC_TIMER_START("step_read_chunks_time");
    //TAU_TRACK_MEMORY_FOOTPRINT();
    //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
#endif
    auto chunks = m_chunk_reader.read_chunks(chunk_id_from, chunk_id_to);
#ifdef TAU_PERF
    TAU_DYNAMIC_TIMER_STOP("step_read_chunks_time");
    TAU_STATIC_TIMER_STOP("total_read_chunks_time");
    //TAU_TRACK_MEMORY_FOOTPRINT();
    //TAU_TRACK_MEMORY_FOOTPRINT_HERE();

    TAU_STATIC_TIMER_START("total_analyze_chunks_time");
    TAU_DYNAMIC_TIMER_START("step_analyze_chunks_time");
#endif
    for (Chunk* chunk : chunks)
    {
        analyze(chunk);
        free_chunk(chunk);
    }
#ifdef TAU_PERF
    TAU_DYNAMIC_TIMER_STOP("step_analyze_chunks_time");
    TAU_STATIC_TIMER_STOP("total_analyze_chunks_time");
#endif
}
