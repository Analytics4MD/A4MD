#include "chunk_reader.h"
//#include <typeinfo>


ChunkReader::ChunkReader(Chunker & chunker)
: m_chunker(chunker)
{
    printf("---===== Created ChunkReader with %s chunker\n",typeid(m_chunker).name());
}

std::vector<Chunk*> ChunkReader::read_chunks(int chunk_id_from, int chunk_id_to)
{
    printf("Calling m_chunker.get_chunks in ChunkReader::read_chunks, chunkidfrom %i, chunkidto %i\n",chunk_id_from, chunk_id_to);
    return m_chunker.get_chunks(chunk_id_from, chunk_id_to);
}
