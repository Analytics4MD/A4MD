#ifndef __MD_RETRIEVER_H__
#define __MD_RETRIEVER_H__
#include "retriever.h"


class MDRetriever : public Retriever
{
    protected:
        int m_n_window_width;
        unsigned long int m_total_chunks;
    public:
        MDRetriever(ChunkStager & chunk_stager, unsigned long int total_chunks, int n_window_width = 1);
        ~MDRetriever();
        void run() override;
};
#endif
