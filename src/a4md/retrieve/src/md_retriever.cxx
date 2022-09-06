#include "../include/md_retriever.h"

MDRetriever::MDRetriever(ChunkStager & chunk_stager, unsigned long int total_chunks, int n_window_width)
: Retriever(chunk_stager),
  m_total_chunks(total_chunks),
  m_n_window_width(n_window_width)
{
    printf("---===== Initialized MDRetriever with total_chunks = %lu and n_window_width = %d\n", m_total_chunks, m_n_window_width);
}

MDRetriever::~MDRetriever()
{
    printf("---===== Finalized MDRetriever\n");
}

void MDRetriever::run()
{
    for(int chunk_id = m_n_window_width; chunk_id < m_total_chunks + m_n_window_width; chunk_id += m_n_window_width)
    {
        auto begin = chunk_id - m_n_window_width;
        auto end = (chunk_id > m_total_chunks) ? m_total_chunks - 1 : chunk_id - 1;
        printf("---===== MDRetriever::run() chunk_from : %d chunk_to : %d\n", begin, end);
        m_chunk_stager.stage_chunks(begin, end);
    } 
}
