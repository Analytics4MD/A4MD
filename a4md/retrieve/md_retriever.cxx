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
    printf("---==== Entering MDRetriever::run() n_frames : %i\n",m_n_frames);
    for (int frame_id=0; frame_id < m_n_frames; frame_id++)
    {
        if (frame_id % m_n_analysis_stride == 0)
        {
            printf("Calling m_chunk_analyzer.analyze_chunks in MDRetriever::run, frame_id:%i\n",frame_id);
            m_chunk_analyzer.analyze_chunks(frame_id, frame_id);
        }
    } 
}
