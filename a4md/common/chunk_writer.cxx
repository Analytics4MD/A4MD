#include "chunk_writer.h"


ChunkWriter::~ChunkWriter()
{
    printf("(%s): finalized\n", __func__);
}
