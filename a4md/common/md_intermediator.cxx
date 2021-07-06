#include "md_intermediator.h"
#include "md_runner.h"

MDIntermediator::MDIntermediator(char* module_name, char* function_name, char* py_path)
{
	m_py_runner = new MDRunner(module_name, function_name, py_path);
	printf("(%s): initialized\n", __func__);
}

MDIntermediator::MDIntermediator(PyRunner* py_runner)
{
    m_py_runner = py_runner;
    printf("(%s): initialized with PyRunner\n", __func__);
}

MDIntermediator::~MDIntermediator()
{
    delete m_py_runner;
    printf("(%s): finalized\n", __func__);
}

std::vector<std::shared_ptr<Chunk>> MDIntermediator::operate_chunks(std::vector<std::shared_ptr<Chunk>> chunks)
{
	std::vector<std::shared_ptr<Chunk>> result;
    printf("(%s): input MDChunks and output other MDChunks\n");
    for(std::shared_ptr<Chunk>& chunk:chunks)
    {
        std::shared_ptr<Chunk> output_chunk = m_py_runner->direct_chunk(chunk);
        if (output_chunk == nullptr)
        {
        	fprintf(stderr, "ERROR: MDIntermediator::operate_chunks One of returned chunks is null pointer\n");
        	return result;
        }
        result.push_back(output_chunk);
    }
    return result;
}
