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
        double m_total_chunk_read_time_ms;
        double m_total_reader_idle_time_ms;
        double *m_step_chunk_read_time_ms;
        double *m_step_reader_idle_time_ms;
        unsigned long int m_wait_ms;
        unsigned long int m_min_wait_ms;
        unsigned long int m_max_wait_ms;
        bool m_count_lost_frames;
        unsigned int m_lost_frames_count; 	
        MPI_Comm m_gcomm;
    public:
        DataSpacesReader(char* var_name, unsigned long int total_chunks, MPI_Comm comm, bool count_lost_frames=false);
        ~DataSpacesReader();
        std::vector<Chunk*> get_chunks(unsigned long int chunks_from, unsigned long int chunks_to) override;

};
#endif
