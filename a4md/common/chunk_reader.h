#ifndef __CHUNK_READER_H__
#define __CHUNK_READER_H__
#include "ims_reader.h"
#include "chunker.h"

class ChunkReader 
{
    private:
        IMSReader* m_ims_reader; 
        Chunker* m_chunker; 
        bool read_from_file;
    public:
        ChunkReader(IMSReader* ims_reader);
        ChunkReader(Chunker* chunker);
        ~ChunkReader();
        ChunkArray read_chunks(int num_chunks);
};
#endif
