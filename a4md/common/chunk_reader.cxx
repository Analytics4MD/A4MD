#include "chunk_reader.h"


ChunkReader::~ChunkReader()
{
    printf("(%s): finalized\n", __func__);
}