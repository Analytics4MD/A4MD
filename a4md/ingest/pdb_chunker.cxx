#include "pdb_chunker.h"

PDBChunker::PDBChunker(PyRunner & py_runner,
                       char* file_path,
                       int position,
                       int natoms)
: m_py_runner(py_runner),
    m_file_path(file_path),
    m_position(position),
    m_natoms(natoms)
{
    printf("---===== Created PDBChunker with file_path = %s, position = %d and natoms = %d\n", m_file_path, m_position, m_natoms);
    // m_next_id = 0;
}

PDBChunker::~PDBChunker()
{
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
    printf("---===== PDBChunker::get_chunks --> Get chunks from chunk_id_from = %lu to chunk_id_to = %lu\n", chunk_id_from, chunk_id_to);
    std::vector<Chunk*> chunks;
    for (unsigned long int chunk_id = chunk_id_from; chunk_id <= chunk_id_from; chunk_id++)
    {
        Chunk* chunk = nullptr;
        int error = m_py_runner.extract_frame(m_file_path, chunk_id, m_position, &chunk, m_natoms);
        if (error == 0 && chunk != nullptr)
        {
            printf("---===== PDBChunker::get_chunks --> Successfully extract frame with chunk_id = %lu\n", chunk_id);
            chunks.push_back(chunk);
        }
        else
        {
            fprintf(stderr, "---===== ERROR: PDBChunker::get_chunks --> Failed to extract frame");
            std::exit(EXIT_FAILURE);
        }
    }
    return chunks;
}
