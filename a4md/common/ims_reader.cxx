#include "ims_reader.h"

std::vector<Chunk*> IMSReader::get_chunks(int num_chunks)
{
    throw NotImplementedException("IMSReader::get_chunks should not be called. It should be overridden in a concrete class\n");
}

std::vector<Chunk*> IMSReader::get_chunks(unsigned long int chunks_from, unsigned long int chunks_to)
{
    throw NotImplementedException("IMSReader::get_chunks should not be called. It should be overridden in a concrete class\n");
}
