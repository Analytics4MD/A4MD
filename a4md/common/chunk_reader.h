#ifndef __CHUNK_READER_H__
#define __CHUNK_READER_H__
#include "chunk.h"

class ChunkReader 
{
    public:
        virtual ~ChunkReader();
        virtual std::vector<Chunk*> read_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to) = 0;
};
#endif
