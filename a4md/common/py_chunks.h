#ifndef __PY_CHUNKS_H__
#define __PY_CHUNKS_H__
#include "chunks.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;


class PyChunk: public Chunk 
{
    public:
        using Chunk::Chunk; // Inherit constructurs
        void print() override { PYBIND11_OVERRIDE(void, Chunk, print, ); }
        void append(std::shared_ptr<Chunk> other_chunk) override { PYBIND11_OVERRIDE(void, Chunk, append, other_chunk); }
};

#endif