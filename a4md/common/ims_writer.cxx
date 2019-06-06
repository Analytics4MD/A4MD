#include "ims_writer.h"

IMSWriter::~IMSWriter()
{
    printf("---===== Finalized IMSWriter\n");
}

void IMSWriter::write_chunks(std::vector<Chunk*> chunks)
{
    throw NotImplementedException("IMSWriter::write_chunks should not be implemented. This should be implemented in a concrete derived class.");
}
