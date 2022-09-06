#ifndef __IMS_READER_H__
#define __IMS_READER_H__
#include "../../common/include/chunk.h"

// Read chunks from an IMS. No application logic here.
class IMSReader
{
    public:
    	virtual ~IMSReader();
        virtual std::vector<Chunk*> get_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to) = 0;
};
#endif
