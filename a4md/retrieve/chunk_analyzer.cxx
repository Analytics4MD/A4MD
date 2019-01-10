#include "chunk_analyzer.h"
#include <typeinfo>


ChunkAnalyzer::ChunkAnalyzer(ChunkReader & chunk_reader)
: m_chunk_reader(chunk_reader)
{
}

void ChunkAnalyzer::analyze(Chunk* chunk)
{
    throw NotImplementedException("This should not be called. override analyze in the concrete function!");
}

void ChunkAnalyzer::analyze_chunks(int chunk_id_from, int chunk_id_to)
{
    //printf("calling m_chunk_reader.read_chunks in ChunkAnalyzer::analyze_chunks, num_chunks: %i, chunk_reader: %s\n",num_chunks,typeid(m_chunk_reader).name()); 
    auto chunks = m_chunk_reader.read_chunks(chunk_id_from, chunk_id_to);
    for (Chunk* chunk : chunks)
    {
        analyze(chunk);
    }
}

