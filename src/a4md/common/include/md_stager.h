#ifndef __MD_STAGER_H__
#define __MD_STAGER_H_
#include "chunk_stager.h"

class MDStager : public ChunkStager
{
    public:
        MDStager(ChunkReader* chunk_reader, ChunkWriter* chunk_writer);
        MDStager(ChunkReader* chunk_reader, ChunkOperator* chunk_operator, ChunkWriter* chunk_writer);
        MDStager(ChunkReader* chunk_reader, std::vector<ChunkOperator*> chunk_operators, ChunkWriter* chunk_writer);
        ~MDStager();

        void free_chunk(Chunk* chunk) override;
        void free_chunks(std::vector<Chunk*> chunks) override;
};

#endif
