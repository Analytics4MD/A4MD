#ifndef __IMS_READER_H__
#define __IMS_READER_H__
#include "chunk.h"

// Read chunks from an IMS. No application logic here.
class IMSReader 
{
    public:
        IMSReader();
    	~IMSReader();
    	virtual void initialize();
    	virtual void finalize();
        virtual std::vector<Chunk> get_chunks(int num_chunks) = 0;
};

class DataspacesReader : public IMSReader 
{
	public:
		DataspacesReader();
		~DataspacesReader();
		void initialize();
		void finalize();
		std::vector<Chunk> get_chunks(int num_chunks);
};

#endif
