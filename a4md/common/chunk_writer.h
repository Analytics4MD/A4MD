#ifndef __CHUNK_WRITER_H__
#define __CHUNK_WRITER_H__
#include "ims_writer.h"

class ChunkWriter // Writes chunks into an IMS. No application logic here.
{
    private:
        IMSWriter* m_ims_writer;
    public:
        bool write_chunks(ChunkArray chunks);
};
#endif
