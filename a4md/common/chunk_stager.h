#ifndef __CHUNK_STAGER_H__
#define __CHUNK_STAGER_H__
#include "chunk_writer.h"
#include "chunk_reader.h"
#include "chunk_operator.h"

class ChunkStager
{
    protected:
        ChunkReader* m_chunk_reader;
        std::vector<ChunkOperator*> m_chunk_operators;
        ChunkWriter* m_chunk_writer;
    public:
        ChunkStager(ChunkReader* chunk_reader, std::vector<ChunkOperator*> chunk_operators, ChunkWriter* chunk_writer);
        virtual ~ChunkStager();

        bool stage_chunks(int num_chunks=1);
        void stage_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to);

        virtual void free_chunk(Chunk* chunk) = 0;
        virtual void free_chunks(std::vector<Chunk*> chunks) = 0;
};

#endif
