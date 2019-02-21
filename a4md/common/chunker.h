#ifndef __CHUNKER_H__
#define __CHUNKER_H__
#include <vector>
#include <queue>
#include <string>
#include "chunk.h"
#include <chrono>
#define timeNow() std::chrono::high_resolution_clock::now()
typedef std::chrono::high_resolution_clock::time_point TimeVar;
typedef std::chrono::duration<double, std::milli> DurationMilli;


class Chunker 
{
    protected:
        std::queue<Chunk*> m_chunkq;
        unsigned long int m_next_id;
    public:
        void append_chunk(Chunk* chunk);
        std::vector<Chunk*> get_chunks(int num_chunks);
        virtual std::vector<Chunk*> get_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to);
};

#endif
