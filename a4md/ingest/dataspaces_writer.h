#ifndef __DATASPACES_WRITER_H__
#define __DATASPACES_WRITER_H__
#include "ims_writer.h"
#include "mpi.h"
#include "plumed_chunker.h"

class DataSpacesWriter : public IMSWriter
{
    private:
        std::string m_var_name;
        MPI_Comm m_gcomm;
    public:
        DataSpacesWriter(char* var_name);
        void write_chunks(std::vector<Chunk> chunks);
        void write_chunks(PlumedChunker chunker);
};
#endif
