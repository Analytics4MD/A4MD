#include "chunk_reader.h"

ChunkReader::ChunkReader(IMSReader* ims_reader)
{
	m_ims_reader = ims_reader;
	read_from_file = false;
}

ChunkReader::ChunkReader(Chunker* chunker)
{
	m_chunker = chunker;
	read_from_file = true;
}

ChunkReader::~ChunkReader()
{
}

ChunkArray ChunkReader::read_chunks(int num_chunks)
{
    if (read_from_file)
    {
        auto chunks = m_chunker->chunks_from_file(num_chunks);
        ChunkArray chunk_ary;
        for (auto chunk:chunks)
            chunk_ary.append(&chunk);
        return chunk_ary;
    }
    else
        return m_ims_reader->get_chunks(num_chunks);
}
