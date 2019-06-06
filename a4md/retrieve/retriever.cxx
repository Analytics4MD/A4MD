#include "retriever.h"


Retriever::Retriever(ChunkAnalyzer & chunk_analyzer)
: m_chunk_analyzer(chunk_analyzer)
{
    printf("---===== Intialized Retriever\n");
}

Retriever::~Retriever()
{
    printf("---===== Finalized Retriever\n");
}

void Retriever::run()
{
    printf("Retriever::run() --> Run\n");
    throw NotImplementedException("This should not be called. Override ths function in the concrete class");
}
