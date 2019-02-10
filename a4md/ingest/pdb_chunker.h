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
        
    public:
        PDBChunker(PyRunner & py_runner, char* file_path, int position = 0);
        int extract_chunk();
        int get_position();
};

#endif
