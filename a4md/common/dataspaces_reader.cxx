#include "dataspaces_reader.h"
#include "dataspaces.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/serialization/vector.hpp>
#include <sstream>
#ifdef BUILT_IN_PERF
#include "timer.h"
#endif
#ifdef TAU_PERF
#include <TAU.h>
#endif

DataSpacesReader::DataSpacesReader(char* var_name, unsigned long int total_chunks, MPI_Comm comm)
: m_size_var_name("chunk_size"),
  m_var_name(var_name),
  m_total_chunks(total_chunks)
{
#ifdef BUILT_IN_PERF
    m_total_data_read_time_ms = 0.0;
    m_total_chunk_read_time_ms = 0.0;
    m_total_reader_idle_time_ms = 0.0;
    m_step_chunk_read_time_ms = new double [m_total_chunks];
    m_step_reader_idle_time_ms = new double [m_total_chunks];
    m_step_size_read_time_ms = new double [m_total_chunks];
    m_step_between_read_time_ms = new double [m_total_chunks];
#endif
    m_gcomm = comm;
    MPI_Barrier(m_gcomm);
    int nprocs;
    MPI_Comm_size(m_gcomm, &nprocs);
    // Initalize DataSpaces
    // # of Peers, Application ID, ptr MPI comm, additional parameters
    // # Peers: Number of connecting clients to the DS server
    // Application ID: Unique idenitifier (integer) for application
    // Pointer to the MPI Communicator, allows DS Layer to use MPI barrier func
    // Addt'l parameters: Placeholder for future arguments, currently NULL.
    dspaces_init(nprocs, 2, &m_gcomm, NULL);
    printf("---===== Initialized dspaces client in DataSpacesReader, var_name : %s, total_chunks: %u \n", var_name, total_chunks);
}

std::vector<Chunk*> DataSpacesReader::get_chunks(unsigned long int chunks_from, unsigned long int chunks_to)
{
#ifdef BUILT_IN_PERF
    TimeVar t_start = timeNow();
#endif
    unsigned long int chunk_id;
    printf("---===== DataSpacesReader::get_chunks with chunk_from %lu, chunk_to %lu\n",chunks_from, chunks_to);
    std::vector<Chunk*> chunks; 
    MPI_Barrier(m_gcomm);
    int ndim = 1;
    uint64_t lb[1] = {0}, ub[1] = {0};
    for (chunk_id = chunks_from; chunk_id<=chunks_to; chunk_id++)
    {
        std::size_t chunk_size;
#ifdef BUILT_IN_PERF
        TimeVar t_istart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_read_idle_time");
        TAU_DYNAMIC_TIMER_START("read_idle_time");
#endif
        dspaces_lock_on_read("size_lock", &m_gcomm);
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("read_idle_time");
        TAU_STATIC_TIMER_STOP("total_read_idle_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli reader_idle_time_ms = timeNow()-t_istart;
        m_step_reader_idle_time_ms[chunk_id] = reader_idle_time_ms.count();
        m_total_reader_idle_time_ms += m_step_reader_idle_time_ms[chunk_id];

        TimeVar t_rsstart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_read_size_time");
        TAU_DYNAMIC_TIMER_START("read_size_time");
#endif
        int error = dspaces_get(m_size_var_name.c_str(),
                                chunk_id,
                                sizeof(std::size_t),
                                ndim,
                                lb,
                                ub,
                                &chunk_size);
        if (error != 0)
            printf("----====== ERROR (%i): Did not read SIZE of chunk id: %i from dataspaces successfully\n",error, chunk_id);
        //    printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        //else

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("read_size_time");
        TAU_STATIC_TIMER_STOP("total_read_size_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli size_read_time_ms = timeNow() - t_rsstart;
        m_step_size_read_time_ms[chunk_id] = size_read_time_ms.count();
#endif 
        dspaces_unlock_on_read("size_lock", &m_gcomm);
        //printf("chunk size read from ds for chunkid %i : %u\n", chunk_id, chunk_size);

        char *input_data = new char [chunk_size];
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_between_read_time");
        TAU_DYNAMIC_TIMER_START("between_read_time");
#endif
#ifdef BUILT_IN_PERF
        TimeVar t_rbstart = timeNow();
#endif
        dspaces_lock_on_read("my_test_lock", &m_gcomm);
#ifdef BUILT_IN_PERF
        DurationMilli between_read_time_ms = timeNow() - t_rbstart;
        m_step_between_read_time_ms[chunk_id] = between_read_time_ms.count();
        TimeVar t_rcstart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("between_read_time");
        TAU_STATIC_TIMER_STOP("total_between_read_time");
        
        TAU_STATIC_TIMER_START("total_read_chunk_time");
        TAU_DYNAMIC_TIMER_START("read_chunk_time");
        TAU_TRACK_MEMORY_FOOTPRINT();
        TAU_TRACK_MEMORY_FOOTPRINT_HERE();
#endif
        error = dspaces_get(m_var_name.c_str(),
                            chunk_id,
                            chunk_size,
                            ndim,
                            lb,
                            ub,
                            input_data);

        if (error != 0)
            printf("----====== ERROR (%i): Did not read chunkid %i from dataspaces successfully\n",error, chunk_id);
        //else
        //    printf("Read chunk id %i from dataspacess successfull\n",chunk_id);
        
#ifdef BUILT_IN_PERF
        DurationMilli read_chunk_time_ms = timeNow()-t_rcstart;
        m_step_chunk_read_time_ms[chunk_id] = read_chunk_time_ms.count();
        m_total_chunk_read_time_ms += m_step_chunk_read_time_ms[chunk_id];
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("read_chunk_time");
        TAU_STATIC_TIMER_STOP("total_read_chunk_time");
#endif
        dspaces_unlock_on_read("my_test_lock", &m_gcomm);

        //printf("Read char array from dataspace:\n %s\n",input_data);
        SerializableChunk chunk;
        std::string instr(input_data);
        std::istringstream iss(instr);//oss.str());
        {
            boost::archive::text_iarchive ia(iss);
            ia >> chunk;
        }
  
        //printf("----===== Read chunk array ");
        //chunks.print();
        chunks.push_back(chunk.get_chunk());
        delete[] input_data;
    }
    //MPI_Barrier(m_gcomm);
#ifdef BUILT_IN_PERF
    DurationMilli read_time_ms = timeNow()-t_start;
    m_total_data_read_time_ms += read_time_ms.count();
    if (chunk_id-1 == m_total_chunks-1)
    {
        printf("total_data_read_time_ms : %f\n",m_total_data_read_time_ms);
        printf("total_chunk_read_time_ms : %f\n",m_total_chunk_read_time_ms);
        printf("total_reader_idle_time_ms : %f\n",m_total_reader_idle_time_ms);
        printf("total_chunks read : %u\n",m_total_chunks);
        printf("step_chunk_read_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_chunk_read_time_ms[step]);
        }
        printf("\n");
        printf("step_reader_idle_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_reader_idle_time_ms[step]);
        }
        printf("\n");
        printf("step_size_read_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_size_read_time_ms[step]);
        }
        printf("\n");
        printf("step_between_read_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_between_read_time_ms[step]);
        }
        printf("\n");
        
        //ToDo: delete in destructor
        delete[] m_step_chunk_read_time_ms;
        delete[] m_step_reader_idle_time_ms;
        delete[] m_step_size_read_time_ms;
        delete[] m_step_between_read_time_ms;
    }
#endif
    return chunks;
}

DataSpacesReader::~DataSpacesReader()
{
    MPI_Barrier(m_gcomm);
    dspaces_finalize();
    printf("---===== Finalized dspaces client in DataSpacesReader\n");
}
