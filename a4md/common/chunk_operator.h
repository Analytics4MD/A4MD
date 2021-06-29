#ifndef __CHUNK_OPERATOR_H__
#define __CHUNK_OPERATOR_H__
#include "chunk.h"

class ChunkOperator 
{
    public:
        virtual ~ChunkOperator();
        virtual std::vector<std::shared_ptr<Chunk>> operate_chunks(std::vector<std::shared_ptr<Chunk>> chunks) = 0;
};
#endif
