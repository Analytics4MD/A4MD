#ifndef __MD_ANALYZER_H__
#define __MD_ANALYZER_H__
#include "chunk_writer.h"
#include "py_runner.h"


class MDAnalyzer : public ChunkWriter
{
    private:
        PyRunner *m_py_runner;
    public:
        MDAnalyzer(char* module_name, char* function_name, char* py_path = (char*)"");
        ~MDAnalyzer();
        void write_chunks(std::vector<Chunk*> chunks) override;
};
#endif
