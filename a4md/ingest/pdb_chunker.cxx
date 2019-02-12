#include "pdb_chunker.h"

PDBChunker::PDBChunker(PyRunner & py_runner,
                       char* file_path,
                       int position)
: m_py_runner(py_runner),
    m_file_path(file_path),
    m_position(position)
{
    printf("---===== Created PDBChunker with file_path = %s and position = %d \n", m_file_path, m_position);
    m_next_id = 0;
}

int PDBChunker::get_position() 
{
    return m_position;
}

int PDBChunker::extract_chunk()
{   
    Chunk* chunk;
    int result  = m_py_runner.extract_frame(m_file_path, m_next_id, m_position, chunk);
    if (result == 0 && chunk != NULL)
    {
        append_chunk(chunk);
    } 
    else 
    {
        fprintf(stderr, "PDBChunker::extract_chunk is not able to extract chunk\n");
    }
    return result;
}
