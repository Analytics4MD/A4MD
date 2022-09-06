#ifndef __CHUNK_ANALYZER_H__
#define __CHUNK_ANALYZER_H__
#include "../../common/include/chunk_reader.h"

// interface for all analyzers
class ChunkAnalyzer
{
    protected:
        ChunkReader & m_chunk_reader;
    public:
        ChunkAnalyzer(ChunkReader & chunk_reader);
        virtual ~ChunkAnalyzer();
        virtual void analyze(Chunk* chunk) = 0;
        virtual void free_chunk(Chunk* chunk) = 0;
        void analyze_chunks(int chunk_id_from, int chunk_id_to);
};
#endif
