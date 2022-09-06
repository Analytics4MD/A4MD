#include "../include/md_intermediator.h"
#include "../include/md_runner.h"

MDIntermediator::MDIntermediator(char* module_name, char* function_name, char* py_path)
{
	m_py_runner = new MDRunner(module_name, function_name, py_path);
	printf("---===== Initialized MDIntermediator\n");
}

MDIntermediator::MDIntermediator(PyRunner* py_runner)
{
    m_py_runner = py_runner;
    printf("---===== Initialized MDIntermediator with initialized PyRunner\n");
}

MDIntermediator::~MDIntermediator()
{
    delete m_py_runner;
    printf("---===== Finalized MDIntermediator\n");
}

std::vector<Chunk*> MDIntermediator::operate_chunks(std::vector<Chunk*> chunks)
{
	std::vector<Chunk*> result;
    printf("---===== MDIntermediator::operate_chunks() Input MDChunks and output other MDChunks\n");
    for(Chunk* chunk:chunks)
    {
        Chunk* output_chunk = m_py_runner->direct_chunk(chunk);
        if (output_chunk == nullptr)
        {
        	fprintf(stderr, "ERROR: MDIntermediator::operate_chunks One of returned chunks is null pointer\n");
        	return result;
        }
        result.push_back(output_chunk);
    }
    return result;
}
