#include "chunk_writer.h"
#ifdef TAU_PERF
#include <TAU.h>
#endif

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
#ifdef TAU_PERF
    TAU_TRACK_MEMORY_FOOTPRINT_HERE();
#endif
    m_ims_writer.write_chunks(chunks);
}
