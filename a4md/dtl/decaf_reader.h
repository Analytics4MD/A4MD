#ifndef __DECAF_READER_H__
#define __DECAF_READER_H__
#include "ims_reader.h"
#include "mpi.h"
#include <decaf/decaf.hpp>

class DecafReader : public IMSReader
{
    private:
        decaf::Decaf* m_decaf;
        std::string m_json_file;
        unsigned int m_total_chunks;
#ifdef BUILT_IN_PERF
        double m_total_data_read_time_ms;
        double m_total_chunk_read_time_ms;
        double m_total_reader_idle_time_ms;
        double m_total_deser_time_ms;
        double *m_step_chunk_read_time_ms;
        double *m_step_reader_idle_time_ms;
        double *m_step_size_read_time_ms;
        double *m_step_between_read_time_ms;
        double *m_step_deser_time_ms;
#endif
#ifdef COUNT_LOST_FRAMES
        unsigned long int m_wait_ms;
        unsigned long int m_min_wait_ms;
        unsigned long int m_max_wait_ms;
        unsigned int m_lost_frames_count; 	
        std::vector<unsigned int> m_lost_frames_id;
#endif
        MPI_Comm m_gcomm;
    public:
        DecafReader(std::string json_file, unsigned long int total_chunks, MPI_Comm comm);
        ~DecafReader();
        std::vector<Chunk*> get_chunks(unsigned long int chunks_from, unsigned long int chunks_to) override;

};
#endif
