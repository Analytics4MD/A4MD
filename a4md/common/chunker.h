#ifndef __CHUNKER_H__
#define __CHUNKER_H__
#include <vector>
#include <queue>
#include <string>
#include "chunk.h"


class Chunker
{
    protected:
        std::queue<Chunk*> m_chunkq;
        unsigned long int m_next_id;
    public:
        virtual ~Chunker();
        void append_chunk(Chunk* chunk);
        std::vector<Chunk*> get_chunks(int num_chunks);
        virtual std::vector<Chunk*> get_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to) = 0;
};

#endif
