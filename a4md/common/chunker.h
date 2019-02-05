#ifndef __CHUNKER_H__
#define __CHUNKER_H__
#include <vector>
#include <string>
#include "chunk.h"
#include <chrono>
#define timeNow() std::chrono::high_resolution_clock::now()
typedef std::chrono::high_resolution_clock::time_point TimeVar;
typedef std::chrono::duration<double, std::milli> DurationMilli;


class Chunker 
{
    public:
        virtual std::vector<Chunk*> get_chunks(int num_chunks);
        virtual std::vector<Chunk*> get_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to);

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