#ifndef __CHUNKER_H__
#define __CHUNKER_H__
#include <vector>
#include <string>
#include "chunk.h"


class Chunker 
{
    public:
        virtual std::vector<Chunk*> get_chunks(int num_chunks);
        virtual std::vector<Chunk*> get_chunks(int chunk_id_from, int chunk_id_to);

};

//class PdbChunker : public Chunker
//{
//    private:
//        std::string m_log_path;
//        std::string m_py_path;
//        std::string m_py_script;
//        std::string m_py_def;
//        PyObject *m_py_func;
//    public:
//        PdbChunker(std::string file_path, std::string log_path, std::string py_path, std::string py_script, std::string py_def);
//        ~PdbChunker();
//
//        void initialize() override;
//        void finalize() override;
//        std::vector<Chunk> chunks_from_file(int num_chunks=1);	
//};

#endif
