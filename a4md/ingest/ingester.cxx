#include "ingester.h"

Ingester::Ingester(ChunkStager & chunk_stager)
: m_chunk_stager(chunk_stager)
{
    printf("---===== Initialized Ingester with %s chunk_stager\n", typeid(m_chunk_stager).name());
}

Ingester::~Ingester()
{
    printf("---===== Finalized Ingester\n");
}

void Ingester::run()
{
    printf("Ingester::Run() --> Run\n");
    throw NotImplementedException("This should not be called. Override this function in the concrete class");
}
