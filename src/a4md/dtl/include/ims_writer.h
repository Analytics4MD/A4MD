#ifndef __IMS_WRITER_H__
#define __IMS_WRITER_H__
#include "../../common/include/chunk.h"

// Writes chunks into an IMS. No application logic here.
class IMSWriter
{
    public:
        virtual ~IMSWriter();
        virtual void write_chunks(std::vector<Chunk*> chunks) = 0;
};
#endif
