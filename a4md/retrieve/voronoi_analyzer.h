#ifndef __VORONOI_ANALYZER_H__
#define __VORONOI_ANALYZER_H__
#include "chunk_analyzer.h"
#include "py_voronoi_analyzer.h"


class VoronoiAnalyzer : public ChunkAnalyzer
{
    private:
        PyVoronoiAnalyzer m_py_analyzer;

    public:
        VoronoiAnalyzer(ChunkReader & chunk_reader, std::string module_name, std::string function_name);
        void analyze(Chunk* chunk) override;
};
#endif
