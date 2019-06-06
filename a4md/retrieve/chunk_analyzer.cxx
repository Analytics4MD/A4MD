#include "chunk_analyzer.h"
#include <typeinfo>
//#include <TAU.h>

ChunkAnalyzer::ChunkAnalyzer(ChunkReader & chunk_reader)
: m_chunk_reader(chunk_reader)
{
    printf("---===== Initialized ChunkAnalyzer with %s chunk_reader\n", typeid(m_chunk_reader).name());
}

ChunkAnalyzer::~ChunkAnalyzer()
{
    printf("---===== Finalized ChunkAnalyzer\n");
}

void ChunkAnalyzer::analyze(Chunk* chunk)
{
    throw NotImplementedException("This should not be called. override analyze in the concrete function!");
}

void ChunkAnalyzer::free(Chunk* chunk)
{
    throw NotImplementedException("This should not be called. override free in the concrete function!");
}

void ChunkAnalyzer::analyze_chunks(int chunk_id_from, int chunk_id_to)
{
    //printf("calling m_chunk_reader.read_chunks in ChunkAnalyzer::analyze_chunks, num_chunks: %i, chunk_reader: %s\n",num_chunks,typeid(m_chunk_reader).name()); 
    //TAU_DYNAMIC_TIMER_START("read_chunks");
    //TAU_TRACK_MEMORY_FOOTPRINT();
    //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
    auto chunks = m_chunk_reader.read_chunks(chunk_id_from, chunk_id_to);
    //TAU_DYNAMIC_TIMER_STOP("read_chunks");
    //TAU_DYNAMIC_TIMER_START("analyze_chunks");
    //TAU_TRACK_MEMORY_FOOTPRINT();
    //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
    for (Chunk* chunk : chunks)
    {
        analyze(chunk);
        //TAU_DYNAMIC_TIMER_STOP("analyze_chunks");
        //TAU_DYNAMIC_TIMER_START("delete_chunks");
        //TAU_TRACK_MEMORY_FOOTPRINT();
        //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
        //MDChunk *md_chunk = dynamic_cast<MDChunk *>(chunk);
        free(chunk);
    }
    //delete data;
    //TAU_DYNAMIC_TIMER_STOP("delete_chunks");
}

