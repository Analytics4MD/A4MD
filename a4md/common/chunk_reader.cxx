#include "chunk_reader.h"
//#include <typeinfo>


ChunkReader::ChunkReader(Chunker & chunker)
: m_chunker(chunker)
{
    printf("---===== Created ChunkReader with %s chunker\n",typeid(m_chunker).name());
}

ChunkReader::~ChunkReader()
{
    printf("---===== Finalized ChunkReader\n");
}

std::vector<Chunk*> ChunkReader::read_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to)
{
    printf("ChunkReader::read_chunks() --> Read chunks from chunk_id_from %lu to chunk_id_to %lu\n",chunk_id_from, chunk_id_to);
    return m_chunker.get_chunks(chunk_id_from, chunk_id_to);
}
