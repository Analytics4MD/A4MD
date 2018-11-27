#ifndef __CHUNK_STAGER_H__
#define __CHUNK_STAGER_H__
#include "common.h"
#include "chunk_writer.h"
#include "chunk_reader.h"

class ChunkStager 
{
    private:
        ChunkReader* m_chunk_reader;
        ChunkWriter* m_chunk_writer;

    public:
        ChunkStager(ChunkReader* reader, ChunkWriter* writer);
        ~ChunkStager();
        bool stage_chunks(int num_chunks);
};

#endif
