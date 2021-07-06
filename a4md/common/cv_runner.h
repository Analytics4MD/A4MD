#ifndef __CV_RUNNER_H__
#define __CV_RUNNER_H__
#include "py_runner.h"

class CVRunner : public PyRunner 
{
	private:
	public:
		CVRunner(char* module_name, char* function_name, char* py_path = (char*)"");
		~CVRunner();

		void input_chunk(std::shared_ptr<Chunk> chunk) override;
        std::shared_ptr<Chunk> output_chunk(unsigned long int chunk_id) override;
        std::shared_ptr<Chunk> direct_chunk(std::shared_ptr<Chunk> chunk) override;
};

#endif /* __CV_RUNNER_H__ */