#ifndef __DATASPACES_READER_H__
#define __DATASPACES_READER_H__
#include "ims_reader.h"
#include "mpi.h"

class DataSpacesReader : public IMSReader
{
    private:
        std::string m_var_name;
        MPI_Comm m_gcomm;
    public:
        DataSpacesReader(char* var_name);
        ChunkArray get_chunks(int num_chunks=1);
};
#endif
