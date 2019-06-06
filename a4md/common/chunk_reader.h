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
        ~ChunkReader();
        std::vector<Chunk*> read_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to);
};
#endif
