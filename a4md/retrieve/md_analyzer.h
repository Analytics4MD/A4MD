#ifndef __MD_ANALYZER_H__
#define __MD_ANALYZER_H__
#include "chunk_analyzer.h"
#include "py_runner.h"


class MDAnalyzer : public ChunkAnalyzer
{
    private:
        PyRunner & m_py_runner;
    public:
        MDAnalyzer(ChunkReader & chunk_reader, PyRunner & py_runner);
        ~MDAnalyzer();
        void analyze(Chunk* chunk) override;
        void free_chunk(Chunk* chunk) override;
};
#endif
