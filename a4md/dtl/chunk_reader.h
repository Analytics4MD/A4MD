#ifndef __CHUNK_READER_H__
#define __CHUNK_READER_H__
#include "ims_reader.h"

class ChunkReader 
{
    private:
        IMSReader & m_ims_reader; 
    public:
        ChunkReader(IMSReader & ims_reader);
        ~ChunkReader();
        std::vector<Chunk*> read_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to);
};
#endif
