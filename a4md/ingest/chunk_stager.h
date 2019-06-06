#ifndef __CHUNK_STAGER_H__
#define __CHUNK_STAGER_H__
#include "chunk_writer.h"
#include "chunk_reader.h"

class ChunkStager 
{
    protected:
        ChunkReader & m_chunk_reader;
        ChunkWriter & m_chunk_writer;
    public:
        ChunkStager(ChunkReader & chunk_reader, ChunkWriter & chunk_writer);
        virtual ~ChunkStager();
        bool stage_chunks(int num_chunks=1);
        void stage_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to);
        virtual void free(Chunk* chunk);
};

#endif
