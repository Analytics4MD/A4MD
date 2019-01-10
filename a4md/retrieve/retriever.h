#ifndef __RETRIEVER_H__
#define __RETRIEVER_H__
#include "chunk_analyzer.h"


class Retriever
{
    protected:
        ChunkAnalyzer & m_chunk_analyzer;
    public:
        Retriever(ChunkAnalyzer & chunk_analyzer);
        virtual void run();
};
#endif
