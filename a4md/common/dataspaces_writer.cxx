#include "dataspaces_writer.h"
#include "dataspaces.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/serialization/vector.hpp>
#include <boost/align/align.hpp>
#include <boost/align/is_aligned.hpp>
#include <sstream>
#ifdef BUILT_IN_PERF
#include "timer.h"
#endif
#ifdef TAU_PERF
#include <TAU.h>
#endif

DataSpacesWriter::DataSpacesWriter(char* var_name, unsigned long int total_chunks, MPI_Comm comm)
: m_size_var_name("chunk_size"),
  m_var_name(var_name),
#ifdef BUILT_IN_PERF
  m_total_data_write_time_ms(0.0),
  m_total_chunk_write_time_ms(0.0),
  m_total_writer_idle_time_ms(0.0),
#endif
  m_total_chunks(total_chunks)
{
#ifdef BUILT_IN_PERF
    m_step_chunk_write_time_ms = new double[m_total_chunks];
    m_step_writer_idle_time_ms = new double[m_total_chunks];
    m_step_size_write_time_ms = new double[m_total_chunks];
    m_step_between_write_time_ms = new double[m_total_chunks];
#endif
    m_gcomm = comm;
    MPI_Barrier(m_gcomm);
    int nprocs;
    MPI_Comm_size(m_gcomm, &nprocs);
    // Initalize DataSpaces
    // # of Peers, Application ID, ptr MPI comm, additional parameters
    // # Peers: Number of connecting clienchunk_id to the DS server
    // Application ID: Unique idenitifier (integer) for application
    // Pointer to the MPI Communicator, allows DS Layer to use MPI barrier func
    // Addt'l parameters: Placeholder for future argumenchunk_id, currently NULL.
    printf("Initializing dpsaces\n");
    dspaces_init(nprocs, 1, &m_gcomm, NULL);
    printf("---===== Initialized dspaces client in DataSpacesWriter, var_name: %s, total_chunks: %u\n",m_var_name.c_str(), m_total_chunks);
}

static inline std::size_t round_up_8(std::size_t n)
{
    return (n%8 == 0) ? n : (n/8 + 1)*8;
}

void DataSpacesWriter::write_chunks(std::vector<Chunk*> chunks)
{
#ifdef BUILT_IN_PERF
    TimeVar t_start = timeNow();
#endif
    unsigned long int chunk_id; 
    MPI_Barrier(m_gcomm);
    for(Chunk* chunk:chunks)
    {
        SerializableChunk serializable_chunk = SerializableChunk(chunk);
        std::ostringstream oss;
        {
            boost::archive::text_oarchive oa(oss);
            // write class instance to archive
            oa << serializable_chunk;
        }
        int ndim = 1;
        uint64_t lb[1] = {0}, ub[1] = {0};
        chunk_id = chunk->get_chunk_id();
        std::string data = oss.str();
        std::size_t size = data.length();
        //printf("MAX SIZE of string is %zu \n", data.max_size());
        //printf("chunk size for chunk_id %i is %zu\n",chunk_id,size);
       
        // Padding to multiple of 8 byte
        std::size_t c_size = round_up_8(size);
        char *c_data = new char [c_size];
        strncpy(c_data, data.c_str(), size);
        //std::size_t r_size = data.copy(c_data, size, 0);
        printf("Padded chunk size %zu\n", c_size);
        //printf("Copied chunk size %zu\n", r_size);

        m_total_size += c_size;
#ifdef BUILT_IN_PERF
        TimeVar t_istart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_idle_time");
        TAU_DYNAMIC_TIMER_START("write_idle_time");
#endif
        dspaces_lock_on_write("size_lock", &m_gcomm);
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("write_idle_time");
        TAU_STATIC_TIMER_STOP("total_write_idle_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli writer_idle_time_ms = timeNow()-t_istart;
        m_step_writer_idle_time_ms[chunk_id] = writer_idle_time_ms.count();
        m_total_writer_idle_time_ms += m_step_writer_idle_time_ms[chunk_id];
        TimeVar t_wsstart = timeNow();
#endif

#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_size_time");
        TAU_DYNAMIC_TIMER_START("write_size_time");
#endif
        int error = dspaces_put(m_size_var_name.c_str(),
                                chunk_id,
                                sizeof(std::size_t),
                                ndim,
                                lb,
                                ub,
                                &c_size);

       
        if (error != 0)
            printf("----====== ERROR: Did not write size of chunk id: %lu to dataspaces successfully\n",chunk_id);
        //else
        //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        error = dspaces_put_sync();
        if (error != 0) 
            printf("----====== ERROR: dspaces_put_sync(%s) failed\n", m_size_var_name.c_str());
#ifdef BUILT_IN_PERF
        DurationMilli size_write_time_ms = timeNow() - t_wsstart;
        m_step_size_write_time_ms[chunk_id] = size_write_time_ms.count();
#endif

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("write_size_time");
        TAU_STATIC_TIMER_STOP("total_write_size_time");
#endif
        
        dspaces_unlock_on_write("size_lock", &m_gcomm);
        //printf("writing char array to dataspace:\n %s\n",data.c_str());
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_between_write_time");
        TAU_DYNAMIC_TIMER_START("between_write_time");
#endif

#ifdef BUILT_IN_PERF
        TimeVar t_wbstart = timeNow();
#endif
        dspaces_lock_on_write("my_test_lock", &m_gcomm);
#ifdef BUILT_IN_PERF
        DurationMilli between_write_time_ms = timeNow() - t_wbstart; 
        m_step_between_write_time_ms[chunk_id] = between_write_time_ms.count();
        TimeVar t_wcstart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("between_write_time");
        TAU_STATIC_TIMER_STOP("total_between_write_time");
        
        TAU_STATIC_TIMER_START("total_write_chunk_time");
        TAU_DYNAMIC_TIMER_START("write_chunk_time");
        TAU_TRACK_MEMORY_FOOTPRINT();
        TAU_TRACK_MEMORY_FOOTPRINT_HERE();
#endif
        error = dspaces_put(m_var_name.c_str(),
                            chunk_id,
                            c_size,
                            ndim,
                            lb,
                            ub,
                            c_data);
        if (error != 0)
            printf("----====== ERROR: Did not write chunk id: %i to dataspaces successfully\n",chunk_id);
        //else
        //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        error = dspaces_put_sync();
        if (error != 0)
            printf("----====== ERROR: dspaces_put_sync(%s) failed\n", m_var_name.c_str());
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("write_chunk_time");
        TAU_STATIC_TIMER_STOP("total_write_chunk_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli write_chunk_time_ms = timeNow()-t_wcstart;
        m_step_chunk_write_time_ms[chunk_id] = write_chunk_time_ms.count();
        m_total_chunk_write_time_ms += m_step_chunk_write_time_ms[chunk_id];
#endif
        //printf("Chunk %lu : step_write_chunk_time_ms : %f\n", m_step_chunk_write_time_ms[chunk_id]);
        dspaces_unlock_on_write("my_test_lock", &m_gcomm);
        delete[] c_data;
#ifdef COUNT_LOST_FRAMES   
        dspaces_lock_on_write("last_write_lock", &m_gcomm);
        error = dspaces_put("last_written_chunk",
                            0,
                            sizeof(unsigned long int),
                            ndim,
                            lb,
                            ub,
                            &chunk_id);
        dspaces_unlock_on_write("last_write_lock", &m_gcomm);
#endif    
    }
    //MPI_Barrier(m_gcomm);
#ifdef BUILT_IN_PERF
    DurationMilli write_time_ms = timeNow()-t_start;
    m_total_data_write_time_ms += write_time_ms.count();
    if (chunk_id == m_total_chunks-1)
    {
        printf("total_chunks written : %u\n",m_total_chunks);
        printf("total_chunk_data_written : %u\n",m_total_size);
        printf("total_data_write_time_ms : %f\n",m_total_data_write_time_ms);
        printf("total_chunk_write_time_ms : %f\n",m_total_chunk_write_time_ms);
        printf("total_writer_idle_time_ms : %f\n",m_total_writer_idle_time_ms);
        printf("step_chunk_write_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_chunk_write_time_ms[step]);
        }
        printf("\n");
        printf("step_writer_idle_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_writer_idle_time_ms[step]);
        }
        printf("\n");
        printf("step_size_write_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_size_write_time_ms[step]);
        }
        printf("\n");
        printf("step_between_write_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_between_write_time_ms[step]);
        }
        printf("\n");

        //Free Built-in Performance Variables
        delete[] m_step_chunk_write_time_ms;
        delete[] m_step_writer_idle_time_ms;
        delete[] m_step_size_write_time_ms;
        delete[] m_step_between_write_time_ms;
    }
#endif
}

DataSpacesWriter::~DataSpacesWriter() 
{
    MPI_Barrier(m_gcomm);
    dspaces_finalize();
    printf("---===== Finalized dspaces client in DataSpacesWriter\n");
}
