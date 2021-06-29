#ifndef __DATASPACES_READER_H__
#define __DATASPACES_READER_H__
#include "chunk_reader.h"
#include "mpi.h"

class DataSpacesReader : public ChunkReader
{
    private:
        int m_client_id;
        int m_group_id;
        std::string m_chunk_lock_name;
        std::string m_size_lock_name;
        std::string m_chunk_var_name;
        std::string m_size_var_name;
        unsigned int m_total_chunks;
#ifdef BUILT_IN_PERF
        double m_total_data_read_time_ms;
        double m_total_chunk_read_time_ms;
        double m_total_reader_idle_time_ms;
        double m_total_deser_time_ms;
        double *m_step_data_read_time_ms;
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
        DataSpacesReader(int client_id, int group_id, unsigned long int total_chunks, MPI_Comm comm);
        ~DataSpacesReader();
        std::vector<std::shared_ptr<Chunk>> read_chunks(unsigned long int chunks_from, unsigned long int chunks_to) override;

};
#endif
