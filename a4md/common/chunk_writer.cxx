#include "chunk_writer.h"

ChunkWriter::ChunkWriter(IMSWriter & ims_writer)
: m_ims_writer(ims_writer)
{
    printf("---===== Initialized ChunkWriter with %s IMSWriter\n", typeid(m_ims_writer).name());
}

ChunkWriter::~ChunkWriter()
{
    printf("---===== Finalized ChunkWriter\n");
}

void ChunkWriter::write_chunks(std::vector<Chunk*> chunks)
{
    printf("---===== ChunkWriter::write_chunks --> Write vector of chunks\n");
    m_ims_writer.write_chunks(chunks);
}
