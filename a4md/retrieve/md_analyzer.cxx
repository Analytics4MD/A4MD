#include "md_analyzer.h"
#include "md_runner.h"
#ifdef TAU_PERF
#include <TAU.h>
#endif

MDAnalyzer::MDAnalyzer(char* module_name, char* function_name, char* py_path)
{
    m_py_runner = new MDRunner(module_name, function_name, py_path);
    printf("---===== Initialized MDAnalyzer\n");
}

MDAnalyzer::~MDAnalyzer()
{
    delete m_py_runner;
    printf("---===== Finalized MDAnalyzer\n");
}

void MDAnalyzer::write_chunks(std::vector<std::shared_ptr<Chunk>> chunks)
{
    printf("---===== MDAnalyzer::write_chunks() Analyze MDChunks\n");
    for(auto chunk:chunks)
    {
        m_py_runner->input_chunk(chunk);
    }
}
