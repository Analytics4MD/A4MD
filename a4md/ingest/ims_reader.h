#ifndef __IMS_READER_H__
#define __IMS_READER_H__
#include "common.h"

// Writes chunks into an IMS. No application logic here.
class IMSReader 
{
    public:
        virtual std::vector<Chunk> get_chunks(int num_chunks) = 0;
};
#endif
