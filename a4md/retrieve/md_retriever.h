#ifndef __MD_RETRIEVER_H__
#define __MD_RETRIEVER_H__
#include "retriever.h"


class MDRetriever : public Retriever
{
    protected:
        int m_n_window_width;
        int m_n_frames;
    public:
        MDRetriever(ChunkAnalyzer & chunk_analyzer,
                    int n_frames,
                    int n_window_width=1);
        void run() override;
};
#endif
