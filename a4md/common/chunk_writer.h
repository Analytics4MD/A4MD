#ifndef __CHUNK_WRITER_H__
#define __CHUNK_WRITER_H__
#include "ims_writer.h"

class ChunkWriter // Writes chunks into an IMS. No application logic here.
{
    private:
        IMSWriter & m_ims_writer;
    public:
        ChunkWriter(IMSWriter & ims_writer);
        ~ChunkWriter();
        void write_chunks(std::vector<Chunk*> chunks);
};
#endif
