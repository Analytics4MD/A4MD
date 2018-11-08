#ifndef __CHUNKER_H__
#define __CHUNKER_H__
#include <vector>
#include <string>
#include <Python.h>

#include "common.h"

class Chunker 
{
    protected:
        std::string m_file_path;
    public:
        Chunker();
        ~Chunker();
        virtual std::vector<Chunk> chunks_from_file(int num_chunks) = 0;
};

class PdbChunker : public Chunker
{
    private:
        std::string m_log_path;
        PyObject *m_py_func;
    public:
        PdbChunker(std::string file_path, std::string log_path, std::string py_path, std::string py_script, std::string py_def);
        ~PdbChunker();
        std::vector<Chunk> chunks_from_file(int num_chunks=1);	
};

#endif