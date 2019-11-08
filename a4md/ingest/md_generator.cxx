#include "md_generator.h"


MDGenerator::MDGenerator(ChunkStager & chunk_stager, unsigned long int total_chunks)
: Ingester(chunk_stager),
  m_total_chunks(total_chunks)
{
    printf("---===== Initialized MDGenerator with total_chunks = %lu\n", m_total_chunks);
}

MDGenerator::~MDGenerator()
{
    printf("---===== Finalized MDGenerator\n");
}

void MDGenerator::run()
{
    for (auto step = 0; step < m_total_chunks; step++)
    {
        printf("MDGenerator::run() --> Step = %d\n", step);
        m_chunk_stager.stage_chunks(step, step);
    }
}
