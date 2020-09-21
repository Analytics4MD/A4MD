#ifndef __CV_RUNNER_H__
#define __CV_RUNNER_H__
#include "py_runner.h"

class CVRunner : public PyRunner 
{
	private:
	public:
		CVRunner(char* module_name, char* function_name, char* py_path = (char*)"");
		~CVRunner();

		void input_chunk(Chunk* chunk) override;
        Chunk* output_chunk(unsigned long int chunk_id) override;
        Chunk* direct_chunk(Chunk* chunk) override;
};

#endif /* __CV_RUNNER_H__ */