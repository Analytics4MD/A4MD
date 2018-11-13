#include "chunk_reader.h"

<<<<<<< HEAD
ChunkReader::ChunkReader(std::string file_path)
{
    m_file_path = file_path;
=======
ChunkReader::ChunkReader(IMSReader* ims_reader)
{
	m_ims_reader = ims_reader;
	read_from_file = false;
}

ChunkReader::ChunkReader(Chunker* chunker)
{
	m_chunker = chunker;
	read_from_file = true;
>>>>>>> feat/ingest_library
}

ChunkReader::~ChunkReader()
{
}

<<<<<<< HEAD
std::vector<Chunk> ChunkReader::read_chunks(int num_chunks, bool read_from_file)
{
    if (read_from_file)
        return m_chunker->chunks_from_file(m_file_path, num_chunks);
    else
        return m_ims_reader->get_chunks(num_chunks);
}
=======
std::vector<Chunk> ChunkReader::read_chunks(int num_chunks)
{
    if (read_from_file)
        return m_chunker->chunks_from_file(num_chunks);
    else
        return m_ims_reader->get_chunks(num_chunks);
}
>>>>>>> feat/ingest_library
