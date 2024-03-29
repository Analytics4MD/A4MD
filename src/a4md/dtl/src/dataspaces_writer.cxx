#include "../include/dataspaces_writer.h"
#include "../../common/include/chunk_serializer.h"
#ifdef BUILT_IN_PERF
#include "../../common/include/timer.h"
#endif
#ifdef TAU_PERF
#include <TAU.h>
#endif

DataSpacesWriter::DataSpacesWriter(int client_id, int group_id, unsigned long int total_chunks, MPI_Comm comm)
: m_client_id(client_id),
  m_group_id(group_id),
#ifdef BUILT_IN_PERF
  m_total_data_write_time_ms(0.0),
  m_total_chunk_write_time_ms(0.0),
  m_total_writer_idle_time_ms(0.0),
  m_total_ser_time_ms(0.0),
#endif
  m_total_chunks(total_chunks)
{
#ifdef BUILT_IN_PERF
    m_step_data_write_time_ms = new double[m_total_chunks];
    m_step_chunk_write_time_ms = new double[m_total_chunks];
    m_step_writer_idle_time_ms = new double[m_total_chunks];
    m_step_size_write_time_ms = new double[m_total_chunks];
    m_step_between_write_time_ms = new double[m_total_chunks];
    m_step_ser_time_ms = new double[m_total_chunks];
#endif
    m_gcomm = comm;
    MPI_Barrier(m_gcomm);

    // Append group id to lock names, var names
    std::string group_str = std::to_string(group_id); 
    m_size_var_name = "var_size";
    m_size_var_name.append(group_str);
    m_chunk_var_name = "var_chunk";
    m_chunk_var_name.append(group_str);

    // Initalize DataSpaces
    // # of Peers, Application ID, ptr MPI comm, additional parameters
    // # Peers: Number of connecting clienchunk_id to the DS server
    // Application ID: Unique idenitifier (integer) for application
    // Pointer to the MPI Communicator, allows DS Layer to use MPI barrier func
    // Addt'l parameters: Placeholder for future argumenchunk_id, currently NULL.
    printf("---===== Initializing dpsaces client id %d\n", m_client_id);
    int error = dspaces_init(m_client_id, &m_client);
    if (error != dspaces_SUCCESS) {
        throw DataLayerException("Could not initialize DataSpaces");
    }
    printf("---===== Initialized dspaces client id #%d and group id #%d in DataSpacesWriter, total_chunks: %u\n",m_client_id, m_group_id, m_total_chunks);
}

static inline std::size_t round_up_8(std::size_t n)
{
    return (n%8 == 0) ? n : (n/8 + 1)*8;
}

void DataSpacesWriter::write_chunks(std::vector<Chunk*> chunks)
{
    unsigned long int chunk_id; 
    printf("---===== DataSpacesWriter::write_chunks\n");
    // MPI_Barrier(m_gcomm);
    for(Chunk* chunk:chunks)
    {
        chunk_id = chunk->get_chunk_id();
#ifdef BUILT_IN_PERF
        TimeVar t_start = timeNow();
#endif
        //Boost Binary Serialization
#ifdef BUILT_IN_PERF
        TimeVar t_serstart = timeNow();
#endif       
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_time");
        TAU_DYNAMIC_TIMER_START("step_write_time");
        
        TAU_STATIC_TIMER_START("total_write_ser_time");
        TAU_DYNAMIC_TIMER_START("step_write_ser_time");
#endif
        SerializableChunk serializable_chunk = SerializableChunk(chunk); 
        std::string data;
        ChunkSerializer<SerializableChunk> chunk_serializer;
        bool ret = chunk_serializer.serialize(serializable_chunk, data);
        if (!ret)
        {
            printf("----====== ERROR: Failed to serialize chunk\n");
        }

        std::size_t size = data.size();
        //printf("MAX SIZE of string is %zu \n", data.max_size());
        // printf("Chunk size for chunk_id %i is %zu\n",chunk_id,size);

        // Data padding to resolve GNI alignment error
#ifdef GNI
        // Padding to multiple of 8 byte
        std::size_t c_size = round_up_8(size);
        char *c_data = new char [c_size];
        //strncpy(c_data, data.c_str(), size);
        std::memcpy(c_data, data.c_str(), size);
        // printf("Padded chunk size %zu\n", c_size);
#else
        std::size_t c_size = size;
        char *c_data = (char*)data.data();
#endif /* GNI */

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_ser_time");
        TAU_STATIC_TIMER_STOP("total_write_ser_time");
        
        TAU_DYNAMIC_TIMER_STOP("step_write_time");
        TAU_STATIC_TIMER_STOP("total_write_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli ser_time_ms = timeNow() - t_serstart;
        m_step_ser_time_ms[chunk_id] = ser_time_ms.count();
        m_total_ser_time_ms += m_step_ser_time_ms[chunk_id];
#endif
        printf("Chunk size %zu\n", c_size);
        m_total_size += c_size;
        int ndim = 1;
        uint64_t lb = 0, ub = 0;
#ifdef BUILT_IN_PERF
        TimeVar t_istart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_stall_time");
        TAU_DYNAMIC_TIMER_START("step_write_stall_time");
        
        TAU_STATIC_TIMER_START("total_write_idle_time");
        TAU_DYNAMIC_TIMER_START("step_write_idle_time");
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_idle_time");
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
        TAU_DYNAMIC_TIMER_START("step_write_size_time");
#endif

        int error;
        printf("Writing size of chunk %lu to DataSpaces\n", chunk_id);
#ifdef DTL_DIMES
        error = dspaces_put_local(m_client,
                                  m_size_var_name.c_str(),
                                  chunk_id,
                                  sizeof(std::size_t),
                                  ndim,
                                  &lb,
                                  &ub,
                                  &c_size);
#else
        error = dspaces_put(m_client,
                            m_size_var_name.c_str(),
                            chunk_id,
                            sizeof(std::size_t),
                            ndim,
                            &lb,
                            &ub,
                            &c_size);
#endif
       
        if (error != 0)
            printf("----====== ERROR: Did not write size of chunk id: %lu to dataspaces successfully\n",chunk_id);
        //else
        //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        printf("Successfully wrote size of chunk %lu to DataSpaces\n", chunk_id);

#ifdef BUILT_IN_PERF
        DurationMilli size_write_time_ms = timeNow() - t_wsstart;
        m_step_size_write_time_ms[chunk_id] = size_write_time_ms.count();
#endif

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_size_time");
        TAU_STATIC_TIMER_STOP("total_write_size_time");
#endif
        
        //printf("writing char array to dataspace:\n %s\n",data.c_str());
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_between_time");
        TAU_DYNAMIC_TIMER_START("step_write_between_time");
#endif

#ifdef BUILT_IN_PERF
        TimeVar t_wbstart = timeNow();
#endif
#ifdef BUILT_IN_PERF
        DurationMilli between_write_time_ms = timeNow() - t_wbstart; 
        m_step_between_write_time_ms[chunk_id] = between_write_time_ms.count();
        TimeVar t_wcstart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_between_time");
        TAU_STATIC_TIMER_STOP("total_write_between_time");
        
        TAU_DYNAMIC_TIMER_STOP("step_write_stall_time");
        TAU_STATIC_TIMER_STOP("total_write_stall_time");
        
        TAU_STATIC_TIMER_START("total_write_time");
        TAU_DYNAMIC_TIMER_START("step_write_time");
        
        TAU_STATIC_TIMER_START("total_write_chunk_time");
        TAU_DYNAMIC_TIMER_START("step_write_chunk_time");
        //TAU_TRACK_MEMORY_FOOTPRINT();
        //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
#endif
lb = 0;
ub = c_size-1;

        printf("Writing chunk %lu to DataSpaces\n", chunk_id);
#ifdef DTL_DIMES
        error = dspaces_put_local(m_client,
                                  m_chunk_var_name.c_str(),
                                  chunk_id,
                                  sizeof(char),
                                  ndim,
                                  &lb,
                                  &ub,
                                  c_data);
#else
        error = dspaces_put(m_client,
                            m_chunk_var_name.c_str(),
                            chunk_id,
                            sizeof(char),
                            ndim,
                            &lb,
                            &ub,
                            c_data);
#endif
        if (error != 0)
            printf("----====== ERROR: Did not write chunk id: %i to dataspaces successfully\n",chunk_id);
        //else
        //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        printf("Successfully wrote chunk %lu to DataSpaces\n", chunk_id);

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_chunk_time");
        TAU_STATIC_TIMER_STOP("total_write_chunk_time");
        
        TAU_DYNAMIC_TIMER_STOP("step_write_time");
        TAU_STATIC_TIMER_STOP("total_write_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli write_chunk_time_ms = timeNow()-t_wcstart;
        m_step_chunk_write_time_ms[chunk_id] = write_chunk_time_ms.count();
        m_total_chunk_write_time_ms += m_step_chunk_write_time_ms[chunk_id];
#endif
        //printf("Chunk %lu : step_write_chunk_time_ms : %f\n", m_step_chunk_write_time_ms[chunk_id]);
#ifdef NERSC
        delete[] c_data;
#endif

#ifdef COUNT_LOST_FRAMES
        lb = 0;
        ub = 0;
        error = dspaces_put(m_client,
                            "last_written_chunk",
                            0,
                            sizeof(unsigned long int),
                            ndim,
                            &lb,
                            &ub,
                            &chunk_id);
#endif
#ifdef BUILT_IN_PERF
        DurationMilli write_time_ms = timeNow()-t_start;
        m_step_data_write_time_ms[chunk_id] = write_time_ms.count();
        m_total_data_write_time_ms += m_step_data_write_time_ms[chunk_id];
#endif
    }
    //MPI_Barrier(m_gcomm);
#ifdef BUILT_IN_PERF
    if (chunk_id == m_total_chunks-1)
    {
        printf("total_chunks written : %u\n",m_total_chunks);
        printf("total_chunk_data_written : %u\n",m_total_size);
        printf("total_data_write_time_ms : %f\n",m_total_data_write_time_ms);
        printf("total_chunk_write_time_ms : %f\n",m_total_chunk_write_time_ms);
        printf("total_writer_idle_time_ms : %f\n",m_total_writer_idle_time_ms);
        printf("total_ser_time_ms : %f\n",m_total_ser_time_ms);
        printf("step_data_write_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_data_write_time_ms[step]);
        }
        printf("\n");
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
        printf("step_ser_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_ser_time_ms[step]);
        }
        printf("\n");

        //Free Built-in Performance Variables
        delete[] m_step_data_write_time_ms;
        delete[] m_step_chunk_write_time_ms;
        delete[] m_step_writer_idle_time_ms;
        delete[] m_step_size_write_time_ms;
        delete[] m_step_between_write_time_ms;
        delete[] m_step_ser_time_ms;
    }
#endif
}

DataSpacesWriter::~DataSpacesWriter() 
{
    // MPI_Barrier(m_gcomm);
    dspaces_kill(m_client);
    dspaces_fini(m_client);
    printf("---===== Finalized dspaces client in DataSpacesWriter\n");
}
