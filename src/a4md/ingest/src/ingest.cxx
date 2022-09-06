#include "../include/ingest.h"

Ingest::Ingest(int max_chunk_id)
{
    m_max_chunk_id = max_chunk_id;
}

Ingest::~Ingest()
{
}

void Ingest::initialize()
{
   m_stagers = get_stagers();
}

void Ingest::run()
{
    initialize();
    printf("In Ingest Run\n");
    //throw NotImplementedException();
    int chunk_id = 0;
    while (chunk_id<m_max_chunk_id)
    {
        for(int i=0;i<m_stagers.size();i++)
        {
            m_stagers[i].stage_chunks();
        }
    }
}
