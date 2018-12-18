#include "md_retriever.h"

MDRetriever::MDRetriever(ChunkAnalyzer& chunk_analyzer,
                         int n_frames,
                         int n_window_width)
: Retriever(chunk_analyzer),
  m_n_frames(n_frames),
  m_n_window_width(n_window_width)
{
}

void MDRetriever::run()
{
    printf("---==== Entering MDRetriever::run() n_frames : %d window width : %d\n",m_n_frames, m_n_window_width);
    for(int chunk_id=m_n_window_width; chunk_id<m_n_frames+m_n_window_width; chunk_id+=m_n_window_width)
    {
        auto begin = chunk_id - m_n_window_width;
        auto end = (chunk_id > m_n_frames) ? m_n_frames-1 : chunk_id-1;
        //printf("Analyze chunks from %d to %d\n", begin, end);
        m_chunk_analyzer.analyze_chunks(begin, end);
    } 
}
