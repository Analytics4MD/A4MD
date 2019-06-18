#include "md_retriever.h"
//#include <TAU.h>

MDRetriever::MDRetriever(ChunkAnalyzer& chunk_analyzer,
                         int n_frames,
                         int n_window_width)
: Retriever(chunk_analyzer),
  m_n_frames(n_frames),
  m_n_window_width(n_window_width)
{
    printf("---===== Initialized MDRetriever with n_frames = %d and n_window_width = %d\n", m_n_frames, m_n_window_width);
}

MDRetriever::~MDRetriever()
{
    printf("---===== Finalized MDRetriever\n");
}

void MDRetriever::run()
{
    printf("---==== Entering MDRetriever::run() n_frames : %d window width : %d\n",m_n_frames, m_n_window_width);
    for(int chunk_id=m_n_window_width; chunk_id<m_n_frames+m_n_window_width; chunk_id+=m_n_window_width)
    {
        //TAU_DYNAMIC_TIMER_START("retriever_step");
        //TAU_TRACK_MEMORY_FOOTPRINT();
        //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
        auto begin = chunk_id - m_n_window_width;
        auto end = (chunk_id > m_n_frames) ? m_n_frames-1 : chunk_id-1;
        //printf("Analyze chunks from %d to %d\n", begin, end);
        m_chunk_analyzer.analyze_chunks(begin, end);
        //TAU_DYNAMIC_TIMER_STOP("retriever_step");
    } 
}
