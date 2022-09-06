#include "../include/retriever.h"


Retriever::Retriever(ChunkStager & chunk_stager)
: m_chunk_stager(chunk_stager)
{
    printf("---===== Intialized Retriever\n");
}

Retriever::~Retriever()
{
    printf("---===== Finalized Retriever\n");
}
