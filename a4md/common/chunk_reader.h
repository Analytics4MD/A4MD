#ifndef __CHUNK_READER_H__
#define __CHUNK_READER_H__
#include "ims_reader.h"
#include "chunker.h"

class ChunkReader 
{
    private:
        Chunker & m_chunker; 
    public:
        ChunkReader(Chunker & chunker);
        std::vector<Chunk*> read_chunks(int chunks_from, int chunks_to);
};
#endif
