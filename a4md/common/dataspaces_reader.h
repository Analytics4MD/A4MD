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
        std::vector<Chunk*> get_chunks(int chunks_from, int chunks_to) override;

};
#endif
