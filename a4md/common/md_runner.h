#ifndef __MD_RUNNER_H__
#define __MD_RUNNER_H__
#include "py_runner.h"

class MDRunner : public PyRunner 
{
	private:
		std::string m_trajectory_path;
		int m_position;
		int m_num_atoms;
	public:
		MDRunner(char* module_name, char* function_name, char* py_path = (char*)"");
		MDRunner(char* module_name, char* function_name, char* py_path, char* trajectory_path, int num_atoms, int position = 0);
		~MDRunner();

		void input_chunk(std::shared_ptr<Chunk> chunk) override;
        std::shared_ptr<Chunk> output_chunk(unsigned long int chunk_id) override;
        std::shared_ptr<Chunk> direct_chunk(std::shared_ptr<Chunk> chunk) override;

        int get_position();
        void set_position(int position);

};

#endif /* __MD_RUNNER_H__ */