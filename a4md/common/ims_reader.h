#ifndef __IMS_READER_H__
#define __IMS_READER_H__
#include "chunker.h"

// Read chunks from an IMS. No application logic here.
class IMSReader : public Chunker 
{
    public:
        std::vector<Chunk*> get_chunks(int num_chunks) override;
        std::vector<Chunk*> get_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to) override;
};
#endif
