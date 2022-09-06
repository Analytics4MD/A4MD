#include "../include/md_generator.h"
#include <chrono>
#include <thread>

MDGenerator::MDGenerator(ChunkStager & chunk_stager, unsigned long int total_chunks, int delay_ms)
: Ingester(chunk_stager),
  m_total_chunks(total_chunks),
  m_delay_ms(delay_ms)
{
    printf("---===== Initialized MDGenerator with total_chunks = %lu and delay_ms = %d\n", m_total_chunks, m_delay_ms);
}

MDGenerator::~MDGenerator()
{
    printf("---===== Finalized MDGenerator\n");
}

void MDGenerator::run()
{
    for (auto step = 0; step < m_total_chunks; step++)
    {
        printf("---===== MDGenerator::run() --> Step = %d\n", step);
        m_chunk_stager.stage_chunks(step, step);
        std::this_thread::sleep_for(std::chrono::milliseconds(m_delay_ms));
    }
}
