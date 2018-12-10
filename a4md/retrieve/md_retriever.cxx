#include "md_retriever.h"

MDRetriever::MDRetriever(ChunkAnalyzer& chunk_analyzer,
                         int n_steps,
                         int n_stride,
                         int n_analysis_stride)
: Retriever(chunk_analyzer),
  m_n_steps(n_steps),
  m_n_stride(n_stride),
  m_n_analysis_stride(n_analysis_stride)
{
    m_n_frames = m_n_steps/m_n_stride;
}

void MDRetriever::run()
{
    //printf("---==== Entering MDRetriever::run() n_frames : %i\n",m_n_frames);
    for (int chunk_id=0; chunk_id <= m_n_frames; chunk_id+=m_n_analysis_stride)
    {
        //printf("Calling m_chunk_analyzer.analyze_chunks in MDRetriever::run, chunk_id:%i\n",chunk_id);
        m_chunk_analyzer.analyze_chunks(chunk_id, chunk_id);
    } 
}
