#ifndef __RETRIEVER_H__
#define __RETRIEVER_H__
#include "chunk_stager.h"

class Retriever
{
    protected:
        ChunkStager & m_chunk_stager;
    public:
        Retriever(ChunkStager & chunk_stager);
        virtual ~Retriever();
        virtual void run() = 0;
};
#endif
