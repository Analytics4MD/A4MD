#include "chunk_reader.h"

ChunkReader::ChunkReader(std::string file_path)
{
    m_file_path = file_path;
}

ChunkReader::~ChunkReader()
{
}

std::vector<Chunk> ChunkReader::read_chunks(int num_chunks, bool read_from_file)
{
    if (read_from_file)
        return m_chunker->chunks_from_file(m_file_path, num_chunks);
    else
        return m_ims_reader->get_chunks(num_chunks);
}
