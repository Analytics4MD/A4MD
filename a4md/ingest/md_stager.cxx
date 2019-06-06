#include "md_stager.h"
#include <vector>

MDStager::MDStager(ChunkReader & chunk_reader, ChunkWriter & chunk_writer)
: ChunkStager(chunk_reader, chunk_writer)
{
    printf("---===== Initalized MDStager\n");
}

MDStager::~MDStager()
{
    printf("---===== Finalized MDStager\n");
}

void MDStager::free(Chunk* chunk)
{
    printf("MDStager::free --> Free memory of MDChunk\n");
    delete chunk;
}
