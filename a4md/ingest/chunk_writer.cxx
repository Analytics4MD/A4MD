#include "chunk_writer.h"

bool ChunkWriter::write_chunks(std::vector <Chunk> chunks)
{
    m_ims_writer->write_chunks(chunks);
}
