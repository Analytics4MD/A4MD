#ifndef __PDB_CHUNKER_H__
#define __PDB_CHUNKER_H__
#include "chunk_reader.h"
#include "py_runner.h"

class PDBChunker : public ChunkReader
{
    private:
        PyRunner *m_py_runner;
        std::string m_trajectory_path;
        int m_num_atoms;
        int m_position;
        bool external = false;
    public:
        PDBChunker(char* module_name, char* function_name, char* py_path, char* trajectory_path, int num_atoms, int position = 0);
        PDBChunker(PyRunner *py_runner);
        ~PDBChunker();
        // int extract_chunk();
        int get_position();
        std::vector<Chunk*> read_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to) override;
};

#endif
