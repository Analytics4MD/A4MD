#ifndef __DATASPACES_WRITER_H__
#define __DATASPACES_WRITER_H__
#include "ims_writer.h"
#include "mpi.h"

class DataSpacesWriter : public IMSWriter
{
    private:
        std::string m_var_name;
        std::string m_size_var_name;
        unsigned int m_total_chunks;
        unsigned long int m_total_size = 0; 
        double m_total_data_write_time_ms;
        MPI_Comm m_gcomm;
    public:
        DataSpacesWriter(char* var_name, unsigned long int total_chunks);
        void write_chunks(std::vector<Chunk*> chunks) override;
};
#endif
