#include "retriever.h"


Retriever::Retriever(ChunkAnalyzer & chunk_analyzer)
: m_chunk_analyzer(chunk_analyzer)
{
}

void Retriever::run()
{
    throw NotImplementedException("This should not be called. Override ths function in the concrete class");
}
