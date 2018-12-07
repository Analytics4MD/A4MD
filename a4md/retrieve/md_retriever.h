#ifndef __MD_RETRIEVER_H__
#define __MD_RETRIEVER_H__
#include "retriever.h"


class MDRetriever : public Retriever
{
    protected:
        int m_n_steps;
        int m_n_stride;
        int m_n_analysis_stride;
        int m_n_frames;
    public:
        MDRetriever(ChunkAnalyzer & chunk_analyzer,
                    int n_steps,
                    int n_stride,
                    int n_analysis_stride=1);
        void run() override;
};
#endif
