#include "chunk_writer.h"

void ChunkWriter::write_chunks(std::vector <Chunk> chunks)
{
    m_ims_writer->write_chunks(chunks);
}