#ifndef __CHUNK_READER_H__
#define __CHUNK_READER_H__
#include "common.h"
#include "ims_reader.h"
#include "chunker.h"

class ChunkReader 
{
    private:
        IMSReader* m_ims_reader; 
        Chunker* m_chunker; 
<<<<<<< HEAD
        std::string m_file_path;

    public:
        ChunkReader(std::string file_path);
        ~ChunkReader();
        std::vector<Chunk> read_chunks(int num_chunks, bool read_from_file);
};
#endif
=======
        bool read_from_file;
    public:
        ChunkReader(IMSReader* ims_reader);
        ChunkReader(Chunker* chunker);
        ~ChunkReader();
        std::vector<Chunk> read_chunks(int num_chunks);
};
#endif
>>>>>>> feat/ingest_library
