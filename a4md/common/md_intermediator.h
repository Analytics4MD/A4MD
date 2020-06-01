#ifndef __MD_INTERMEDIATOR_H__
#define __MD_INTERMEDIATOR_H__
#include "chunk_operator.h"
#include "py_runner.h"

class MDIntermediator : public ChunkOperator
{
	private:
		PyRunner *m_py_runner;
    public:
        MDIntermediator(char* module_name, char* function_name, char* py_path = (char*)"");
        ~MDIntermediator();
        std::vector<Chunk*> operate_chunks(std::vector<Chunk*> chunks) override;
};
#endif /* __MD_INTERMEDIATOR_H__ */