#ifndef __DATASPACES_WRITER_H__
#define __DATASPACES_WRITER_H__
#include "ims_writer.h"
#include "mpi.h"
#include <decaf/decaf.hpp>

class DecafWriter : public IMSWriter
{
    private:
        decaf::Decaf* m_decaf;
        std::string m_json_file;
        unsigned int m_total_chunks;
        unsigned long int m_total_size = 0;
        MPI_Comm m_gcomm;
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
    public:
        DecafWriter(std::string json_file, unsigned long int total_chunks, MPI_Comm comm);
        ~DecafWriter();
        void write_chunks(std::vector<Chunk*> chunks) override;
};
#endif
