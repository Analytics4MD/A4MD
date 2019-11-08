#ifndef __MD_GENERATOR_H__
#define __MD_GENERATOR_H__
#include "ingester.h"
#include "chunk_stager.h"

class MDGenerator : public Ingester
{
    private:
        unsigned long int m_total_chunks;
    public:
        MDGenerator(ChunkStager & chunk_stager, unsigned long int total_chunks);
        ~MDGenerator();
        void run() override;
};
#endif
