#ifndef __IMS_READER_H__
#define __IMS_READER_H__
#include <vector>

#include "common.h"

// Read chunks from an IMS. No application logic here.
class IMSReader 
{
    public:
        virtual std::vector<Chunk> get_chunks(int num_chunks) = 0;
};
#endif