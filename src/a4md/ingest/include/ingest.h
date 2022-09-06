#ifndef __INGEST_H__
#define __INGEST_H__
#include <vector>
#include "chunk_stager.h"

class Ingest
{
    private:
        std::vector<ChunkStager> m_stagers;
        int m_max_chunk_id;
        void initialize();
    protected:
        virtual std::vector<ChunkStager> get_stagers() = 0;
    public:
        Ingest(int max_chunk_id);
        ~Ingest();
        void run();
};
#endif
