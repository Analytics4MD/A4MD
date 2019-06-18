#ifndef __INGESTER_H__
#define __INGESTER_H__
#include <vector>
#include "chunk_stager.h"

class Ingester
{
    protected:
        ChunkStager & m_chunk_stager;
    public:
        Ingester(ChunkStager & chunk_stager);
        ~Ingester();
        virtual void run() = 0;
};
#endif
