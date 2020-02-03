#ifndef __DATASPACES_WRITER_H__
#define __DATASPACES_WRITER_H__
#include "ims_writer.h"
#include "mpi.h"

class DataSpacesWriter : public IMSWriter
{
    private:
        int m_client_id;
        std::string m_var_name;
        std::string m_size_var_name;
        unsigned int m_total_chunks;
        unsigned long int m_total_size = 0;
#ifdef BUILT_IN_PERF
        double m_total_data_write_time_ms;
        double m_total_chunk_write_time_ms;
        double m_total_writer_idle_time_ms;
        double m_total_ser_time_ms;
        double *m_step_chunk_write_time_ms;
        double *m_step_writer_idle_time_ms;
        double *m_step_size_write_time_ms;
        double *m_step_between_write_time_ms;
        double *m_step_ser_time_ms;
#endif
        MPI_Comm m_gcomm;
    public:
        DataSpacesWriter(int client_id, char* var_name, unsigned long int total_chunks, MPI_Comm comm);
        ~DataSpacesWriter();
        void write_chunks(std::vector<Chunk*> chunks) override;
};
#endif
