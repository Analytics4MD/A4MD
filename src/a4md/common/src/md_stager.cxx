#include "../include/md_stager.h"
#include <vector>

MDStager::MDStager(ChunkReader* chunk_reader, ChunkWriter* chunk_writer)
: ChunkStager(chunk_reader, std::vector<ChunkOperator*>(), chunk_writer)
{
    printf("---===== Initalized MDStager\n");
}

MDStager::MDStager(ChunkReader* chunk_reader, ChunkOperator* chunk_operator, ChunkWriter* chunk_writer)
: ChunkStager(chunk_reader, std::vector<ChunkOperator*>{chunk_operator}, chunk_writer)
{
    printf("---===== Initalized MDStager\n");
}

MDStager::MDStager(ChunkReader* chunk_reader, std::vector<ChunkOperator*> chunk_operators, ChunkWriter* chunk_writer)
: ChunkStager(chunk_reader, chunk_operators, chunk_writer)
{
	printf("---===== Initalized MDStager\n");
}

MDStager::~MDStager()
{
    printf("---===== Finalized MDStager\n");
}

void MDStager::free_chunk(Chunk* chunk)
{
    printf("---===== MDStager::free_chunk --> Free memory of MDChunk\n");
    delete chunk;
}

void MDStager::free_chunks(std::vector<Chunk*> chunks) 
{
	printf("---===== MDStager::free_chunks --> Free memory of MDChunks\n");
	for (Chunk* chunk : chunks) 
	{
		delete chunk;
	}
}
