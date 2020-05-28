#ifndef __MD_STAGER_H__
#define __MD_STAGER_H_
#include "chunk_stager.h"
#include "chunk_writer.h"
#include "chunk_reader.h"

class MDStager : public ChunkStager
{
    public:
        MDStager(ChunkReader & chunk_reader, ChunkWriter & chunk_writer);
        ~MDStager();
        void free_chunk(Chunk* chunk) override;
};

#endif
