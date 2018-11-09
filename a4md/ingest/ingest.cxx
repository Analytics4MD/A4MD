#include "ingest.h"

Ingest::Ingest()
{
    initialize();
}

Ingest::~Ingest()
{
}

void Ingest::initialize() //called from the constructor
{
   m_stagers = get_stagers();
}

void Ingest::run()
{
    printf("In Ingest Run\n");
    throw NotImplementedException();
    //while (chunkId<maxChunkId)
    //{
    //    for(i=0;i<m_stagers.len();i++)
    //    {
    //        this.m_stagers[i].stage_chunks(num_chunks)
    //    }
    //}
}