#ifndef __IMS_WRITER_H__
#define __IMS_WRITER_H__
#include "chunk.h"

// Writes chunks into an IMS. No application logic here.
class IMSWriter 
{
    public:
        virtual void write_chunks(ChunkArray chunks) = 0;
};
#endif
