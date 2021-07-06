#include "chunk_operator.h"

ChunkOperator::~ChunkOperator()
{
    printf("(%s): finalized\n", __func__);
}