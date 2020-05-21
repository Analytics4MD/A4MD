#ifndef __DECAF_WRITER_H__
#define __DECAF_WRITER_H__
#include "chunk_writer.h"
#include "mpi.h"
#include <decaf/decaf.hpp>
#include <bredala/data_model/pconstructtype.h>
#include <bredala/data_model/arrayfield.hpp>

class DecafWriter : public ChunkWriter
{
    private:
        decaf::Decaf* m_decaf;
        std::string m_json_file;
        unsigned int m_total_chunks;
        unsigned long int m_total_size = 0;
        bool initialized = false;
#ifdef BUILT_IN_PERF
        double m_total_data_write_time_ms;
        double m_total_chunk_write_time_ms;
        double m_total_writer_idle_time_ms;
        double m_total_ser_time_ms;
        double *m_step_data_write_time_ms;
        double *m_step_chunk_write_time_ms;
        double *m_step_writer_idle_time_ms;
        double *m_step_ser_time_ms;
#endif
        MPI_Comm m_gcomm;
    public:
        DecafWriter(std::string json_file, unsigned long int total_chunks, MPI_Comm comm);
        DecafWriter(decaf::Decaf *decaf, unsigned long int total_chunks, MPI_Comm comm);
        ~DecafWriter();
        void write_chunks(std::vector<Chunk*> chunks) override;
};
#endif
