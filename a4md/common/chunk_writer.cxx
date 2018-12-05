#include "chunk_writer.h"

bool ChunkWriter::write_chunks(ChunkArray chunks)
{
    m_ims_writer->write_chunks(chunks);
}
