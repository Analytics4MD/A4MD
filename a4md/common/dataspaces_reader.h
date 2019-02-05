#ifndef __DATASPACES_READER_H__
#define __DATASPACES_READER_H__
#include "ims_reader.h"
#include "mpi.h"

class DataSpacesReader : public IMSReader
{
    private:
        std::string m_var_name;
        std::string m_size_var_name;
        unsigned int m_total_chunks;
        double m_total_data_read_time_ms;
        MPI_Comm m_gcomm;
    public:
        DataSpacesReader(char* var_name, unsigned long int total_chunks, MPI_Comm comm);
        std::vector<Chunk*> get_chunks(unsigned long int chunks_from, unsigned long int chunks_to) override;

};
#endif
