#ifndef __PDB_CHUNKER_H__
#define __PDB_CHUNKER_H__
#include "chunker.h"
#include "py_runner.h"

class PDBChunker : public Chunker
{
    private:
        PyRunner & m_py_runner;
        char* m_file_path;
        int m_position;
        int m_natoms;
    public:
        PDBChunker(PyRunner & py_runner, char* file_path, int position, int natoms = 0);
        ~PDBChunker();
        int extract_chunk();
        int get_position();
        std::vector<Chunk*> get_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to) override;
};

#endif
