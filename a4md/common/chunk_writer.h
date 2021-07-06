#ifndef __CHUNK_WRITER_H__
#define __CHUNK_WRITER_H__
#include "chunk.h"

class ChunkWriter // Writes chunks into an IMS. No application logic here.
{
    public:
        virtual ~ChunkWriter();
        virtual void write_chunks(std::vector<std::shared_ptr<Chunk>> chunks) = 0;
};
#endif
