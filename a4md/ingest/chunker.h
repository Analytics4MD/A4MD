#ifndef __CHUNKER_H__
#define __CHUNKER_H__
#include "common.h"

class Chunker 
{
    public:
        Chunker();
        ~Chunker();
        virtual std::vector<Chunk> chunks_from_file(std::string file_path, int num_chunks) = 0;
};

class PdbChunker : public Chunker
{
    private:
        std::string m_file_path;

    public:
        PdbChunker(std::string file_path);
        ~PdbChunker();
        std::vector<Chunk> chunks_from_file(std::string file_path, int num_chunks=1);	
};

#endif
