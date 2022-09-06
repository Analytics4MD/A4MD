#include "../include/pdb_chunker.h"
#include "../../common/include/md_runner.h"

PDBChunker::PDBChunker(char* module_name, char* function_name, char* py_path, char* trajectory_path, int num_atoms, int position)
: m_trajectory_path(trajectory_path),
  m_num_atoms(num_atoms),
  m_position(position)
{
    m_py_runner = new MDRunner(module_name, function_name, py_path, trajectory_path, num_atoms, position);
    printf("---===== Initialized PDBChunker with trajectory_path = %s, num_atoms = %d and position = %d\n", m_trajectory_path.c_str(), m_num_atoms, m_position);
}

PDBChunker::PDBChunker(PyRunner *py_runner)
: m_py_runner(py_runner)
{
    external = true;
    printf("---===== Initialized PDBChunker with external PyRunner\n");
}

PDBChunker::~PDBChunker()
{
    if (!external)
    {
        delete m_py_runner;
    }
    printf("---===== Finalized PDBChunker\n");
}

int PDBChunker::get_position() 
{
    return m_position;
}

// int PDBChunker::extract_chunk()
// {   
//     Chunk* chunk = nullptr;
//     int result  = m_py_runner.extract_frame(m_file_path, m_next_id, m_position, &chunk, m_natoms);
//     if (result == 0 && chunk != nullptr)
//     {
//         append_chunk(chunk);
//     } 
//     else 
//     {
//         fprintf(stderr, "PDBChunker::extract_chunk is not able to extract chunk\n");
//     }
//     return result;
// }

std::vector<Chunk*> PDBChunker::read_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to)
{
    printf("---===== PDBChunker::read_chunks --> Get chunks from chunk_id_from = %lu to chunk_id_to = %lu\n", chunk_id_from, chunk_id_to);
    std::vector<Chunk*> chunks;
    for (unsigned long int chunk_id = chunk_id_from; chunk_id <= chunk_id_from; chunk_id++)
    {
        Chunk* chunk = nullptr;
        chunk = m_py_runner->output_chunk(chunk_id);
        if (chunk != nullptr)
        {
            printf("---===== PDBChunker::read_chunks --> Successfully extract frame with chunk_id = %lu\n", chunk_id);
            chunks.push_back(chunk);
        }
        else
        {
            fprintf(stderr, "---===== ERROR: PDBChunker::read_chunks --> Failed to extract frame");
            return chunks;
        }
    }
    return chunks;
}
