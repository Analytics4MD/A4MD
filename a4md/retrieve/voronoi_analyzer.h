#ifndef __VORONOI_ANALYZER_H__
#define __VORONOI_ANALYZER_H__
#include "chunk_analyzer.h"
#include "py_voronoi_analyzer.h"


class VoronoiAnalyzer : public ChunkAnalyzer
{
    private:
        PyVoronoiAnalyzer & m_py_analyzer;

    public:
        VoronoiAnalyzer(ChunkReader & chunk_reader, PyVoronoiAnalyzer & py_analyzer);
        void analyze(Chunk* chunk) override;
};
#endif
