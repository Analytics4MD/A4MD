#include "ims_reader.h"

IMSReader::IMSReader() 
{
	initialize();
}

IMSReader::~IMSReader()
{
	finalize();
}


DataspacesReader::DataspacesReader() 
{

}

DataspacesReader::~DataspacesReader()
{

}

void DataspacesReader::initialize()
{

}

void DataspacesReader::finalize()
{

}

std::vector<Chunk> DataspacesReader::get_chunks(int num_chunks)
{
	std::vector<Chunk> chunks;
    // throw NotImplementedException();
    return chunks;
}