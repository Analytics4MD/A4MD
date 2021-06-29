#include "dataspaces_reader.h"
#include "dataspaces.h"
#ifdef DTL_DIMES
#include "dimes_interface.h"
#endif
#include "chunk_serializer.h"
#include <sstream>
#if defined(BUILT_IN_PERF) || defined(COUNT_LOST_FRAMES)
#include "timer.h"
#endif
#ifdef TAU_PERF
#include <TAU.h>
#endif
#ifdef COUNT_LOST_FRAMES
#include <thread>
#endif

DataSpacesReader::DataSpacesReader(int client_id, int group_id, unsigned long int total_chunks, MPI_Comm comm)
: m_client_id(client_id),
  m_group_id(group_id),
#ifdef BUILT_IN_PERF
  m_total_data_read_time_ms(0.0),
  m_total_chunk_read_time_ms(0.0),
  m_total_reader_idle_time_ms(0.0),
  m_total_deser_time_ms(0.0),
#endif
#ifdef COUNT_LOST_FRAMES
  m_wait_ms(1000),// default wait time of 1 second
  m_min_wait_ms(100),// default min wait time of 100 ms
  m_max_wait_ms(30000),//defailt max wait time of 30 seconds
  m_lost_frames_count(0),
#endif
  m_total_chunks(total_chunks)
{
#ifdef BUILT_IN_PERF
    m_step_data_read_time_ms = new double [m_total_chunks];
    m_step_chunk_read_time_ms = new double [m_total_chunks];
    m_step_reader_idle_time_ms = new double [m_total_chunks];
    m_step_size_read_time_ms = new double [m_total_chunks];
    m_step_between_read_time_ms = new double [m_total_chunks];
    m_step_deser_time_ms = new double [m_total_chunks];
#endif
    m_gcomm = comm;
    MPI_Barrier(m_gcomm);
    int nprocs;
    MPI_Comm_size(m_gcomm, &nprocs);

    // Append group id to lock names, var names
    std::string group_str = std::to_string(group_id); 
    m_size_lock_name = "lock_size";
    m_size_lock_name.append(group_str);
    m_chunk_lock_name = "lock_chunk";
    m_chunk_lock_name.append(group_str);
    m_size_var_name = "var_size";
    m_size_var_name.append(group_str);
    m_chunk_var_name = "var_chunk";
    m_chunk_var_name.append(group_str);

    // Initalize DataSpaces
    // # of Peers, Application ID, ptr MPI comm, additional parameters
    // # Peers: Number of connecting clients to the DS server
    // Application ID: Unique idenitifier (integer) for application
    // Pointer to the MPI Communicator, allows DS Layer to use MPI barrier func
    // Addt'l parameters: Placeholder for future arguments, currently NULL.
    printf("---===== Initializing dpsaces client id %d\n", m_client_id);
    dspaces_init(nprocs, m_client_id, &m_gcomm, NULL);
    printf("---===== Initialized dspaces client id #%d in DataSpacesReader, total_chunks: %u \n", m_client_id, m_total_chunks);
}

std::vector<std::shared_ptr<Chunk>> DataSpacesReader::read_chunks(unsigned long int chunks_from, unsigned long int chunks_to)
{
    unsigned long int chunk_id;
    printf("---===== DataSpacesReader::read_chunks with chunk_from %lu, chunk_to %lu\n",chunks_from, chunks_to);
    std::vector<std::shared_ptr<Chunk>> chunks; 
    MPI_Barrier(m_gcomm);
    int ndim = 1;
    uint64_t lb[1] = {0}, ub[1] = {0};
    for (chunk_id = chunks_from; chunk_id<=chunks_to; chunk_id++)
    {
#ifdef BUILT_IN_PERF
        TimeVar t_start = timeNow();
#endif
#ifdef COUNT_LOST_FRAMES
        unsigned long int last_chunk_id = 0;
        bool try_next_chunk = false;
        auto temp_chunk_id = chunk_id;
        while (last_chunk_id < temp_chunk_id)
        {	       
            dspaces_lock_on_read("last_write_lock", &m_gcomm);
            int error = dspaces_get("last_written_chunk",
                                0,
                                sizeof(unsigned long int),
                                ndim,
                                lb,
                                ub,
                                &last_chunk_id);
            dspaces_unlock_on_read("last_write_lock", &m_gcomm);
            int dchunk = last_chunk_id-temp_chunk_id;
            if (error != -11)
            {
                if (dchunk == 0)
                {
                    //printf("---=== last chunk id as expected. Reading %lu\n",chunk_id);
                    m_wait_ms = std::max(m_min_wait_ms, m_wait_ms / 2);
                }
                else if(dchunk > 0)
                {
                    //printf("---=== lost %i frames before %lu. Current chunk is %lu\n",dchunk,last_chunk_id,chunk_id);
                    try_next_chunk = true;
                    m_wait_ms = std::min(m_max_wait_ms, m_wait_ms * 2);
                    break;
                }
                else
                {
                    //printf("Dont have chunk %lu yet. So waiting %lu ms to poll\n",chunk_id,m_wait_ms);
                    std::this_thread::sleep_for(std::chrono::milliseconds(m_wait_ms));
                }
            }
            else
            {
                printf("---=== Something went terribly wrong while trying to get chunk: %lu \n",chunk_id);
                break;
            }
        }
        if (try_next_chunk)
        {
            m_lost_frames_count++;
            m_lost_frames_id.push_back(chunk_id);
            continue;
        }
#endif /* COUNT_LOST_FRAMES */
	
        std::size_t chunk_size;
#ifdef BUILT_IN_PERF
        TimeVar t_istart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_read_stall_time");
        TAU_DYNAMIC_TIMER_START("step_read_stall_time");

        TAU_STATIC_TIMER_START("total_read_idle_time");
        TAU_DYNAMIC_TIMER_START("step_read_idle_time");
#endif
        dspaces_lock_on_read(m_size_lock_name.c_str(), &m_gcomm);
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_read_idle_time");
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
        TAU_DYNAMIC_TIMER_START("step_read_size_time");
#endif
        int error;
#ifdef DTL_DIMES
        error = dimes_get(m_size_var_name.c_str(),
                                chunk_id,
                                sizeof(std::size_t),
                                ndim,
                                lb,
                                ub,
                                &chunk_size);
#else
        error = dspaces_get(m_size_var_name.c_str(),
                                chunk_id,
                                sizeof(std::size_t),
                                ndim,
                                lb,
                                ub,
                                &chunk_size);
#endif
        if (error != 0)
        {
            if (error == -11)
            {
#ifdef COUNT_LOST_FRAMES
                printf("Recieved -11 from dspaces get. Probably lost chunk %lu\n",chunk_id);
                dspaces_unlock_on_read("size_lock", &m_gcomm);
                m_lost_frames_count++;
                m_lost_frames_id.push_back(chunk_id);
                continue;
#endif /* COUNT_LOST_FRAMES */
                throw DataLayerException("Dataspaces get recieved error code -11. This is not expected for lock type 2, but expected for lock type 1 or 3. Check lock type used.\n");
            } else
                printf("----====== ERROR (%i): Did not read SIZE of chunk id: %lu from dataspaces successfully\n",error, chunk_id); 
        }
        //else
        //    printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        // printf("chunk_size =  %i for chunk id %i\n",chunk_size, chunk_id);

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_read_size_time");
        TAU_STATIC_TIMER_STOP("total_read_size_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli size_read_time_ms = timeNow() - t_rsstart;
        m_step_size_read_time_ms[chunk_id] = size_read_time_ms.count();
#endif 
        dspaces_unlock_on_read(m_size_lock_name.c_str(), &m_gcomm);
        //printf("chunk size read from ds for chunkid %i : %u\n", chunk_id, chunk_size);
        char *input_data = new char [chunk_size];
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_read_between_time");
        TAU_DYNAMIC_TIMER_START("step_read_between_time");
#endif
#ifdef BUILT_IN_PERF
        TimeVar t_rbstart = timeNow();
#endif
        dspaces_lock_on_read(m_chunk_lock_name.c_str(), &m_gcomm);
#ifdef BUILT_IN_PERF
        DurationMilli between_read_time_ms = timeNow() - t_rbstart;
        m_step_between_read_time_ms[chunk_id] = between_read_time_ms.count();
        TimeVar t_rcstart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_read_between_time");
        TAU_STATIC_TIMER_STOP("total_read_between_time");
        
        TAU_DYNAMIC_TIMER_STOP("step_read_stall_time");
        TAU_STATIC_TIMER_STOP("total_read_stall_time");
        
        TAU_STATIC_TIMER_START("total_read_time");
        TAU_DYNAMIC_TIMER_START("step_read_time");
        
        TAU_STATIC_TIMER_START("total_read_chunk_time");
        TAU_DYNAMIC_TIMER_START("step_read_chunk_time");
        //TAU_TRACK_MEMORY_FOOTPRINT();
        //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
#endif

#ifdef DTL_DIMES
        error = dimes_get(m_chunk_var_name.c_str(),
                            chunk_id,
                            chunk_size,
                            ndim,
                            lb,
                            ub,
                            input_data);
#else
        error = dspaces_get(m_chunk_var_name.c_str(),
                            chunk_id,
                            chunk_size,
                            ndim,
                            lb,
                            ub,
                            input_data);
#endif
        if (error != 0)
        {
            if (error == -11)
            {
                printf("Recieved -11 from dspaces get. Probably lost chunk %lu\n",chunk_id);
#ifdef COUNT_LOST_FRAMES
                dspaces_unlock_on_read("my_test_lock", &m_gcomm);
                m_lost_frames_count++;
                m_lost_frames_id.push_back(chunk_id);
                delete [] input_data;
                continue;
#endif /* COUNT_LOST_FRAMES */
                throw DataLayerException("Dataspaces get recieved error code -11. This is not expected for lock type 2, but expected for lock type 1 or 3. Check lock type used.\n");
            }
            else 
                printf("----====== ERROR (%i): Did not read chunkid %lu from dataspaces successfully\n",error, chunk_id);
        }
        //else
        //    printf("Read chunk id %i from dataspacess successfull\n",chunk_id);
        
#ifdef BUILT_IN_PERF
        DurationMilli read_chunk_time_ms = timeNow()-t_rcstart;
        m_step_chunk_read_time_ms[chunk_id] = read_chunk_time_ms.count();
        m_total_chunk_read_time_ms += m_step_chunk_read_time_ms[chunk_id];
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_read_chunk_time");
        TAU_STATIC_TIMER_STOP("total_read_chunk_time");
#endif
        dspaces_unlock_on_read(m_chunk_lock_name.c_str(), &m_gcomm);
       
        // Boost Binary Serialization
#ifdef BUILT_IN_PERF
        TimeVar t_deserstart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_read_deser_time");
        TAU_DYNAMIC_TIMER_START("step_read_deser_time");
#endif
        SerializableChunk serializable_chunk;
        ChunkSerializer<SerializableChunk> chunk_serializer;
        bool ret = chunk_serializer.deserialize(serializable_chunk, input_data, chunk_size);
        if (!ret)
        {
            printf("----====== ERROR: Failed to deserialize chunk\n");
        }
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_read_deser_time");
        TAU_STATIC_TIMER_STOP("total_read_deser_time");
        
        TAU_DYNAMIC_TIMER_STOP("step_read_time");
        TAU_STATIC_TIMER_STOP("total_read_time");
#endif

#ifdef BUILT_IN_PERF
        DurationMilli deser_time_ms = timeNow() - t_deserstart;
        m_step_deser_time_ms[chunk_id] = deser_time_ms.count();
        m_total_deser_time_ms += m_step_deser_time_ms[chunk_id];
#endif

        //printf("----===== Read chunk array ");
        //chunks.print();
        chunks.push_back(serializable_chunk.get_chunk());
        delete[] input_data;
#ifdef BUILT_IN_PERF
        DurationMilli read_time_ms = timeNow()-t_start;
        m_step_data_read_time_ms[chunk_id] = read_time_ms.count();
        m_total_data_read_time_ms += m_step_data_read_time_ms[chunk_id];
#endif
    }
    //MPI_Barrier(m_gcomm);
#ifdef BUILT_IN_PERF
    if (chunk_id-1 == m_total_chunks-1)
    {
        printf("total_chunks read : %u\n",m_total_chunks);
        printf("total_data_read_time_ms : %f\n",m_total_data_read_time_ms);
        printf("total_chunk_read_time_ms : %f\n",m_total_chunk_read_time_ms);
        printf("total_reader_idle_time_ms : %f\n",m_total_reader_idle_time_ms);
        printf("total_deser_time_ms : %f\n",m_total_deser_time_ms);
        printf("step_data_read_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_data_read_time_ms[step]);
        }
        printf("\n");
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
        printf("step_deser_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_deser_time_ms[step]);
        }
        printf("\n");

        //ToDo: delete in destructor
        delete[] m_step_data_read_time_ms;
        delete[] m_step_chunk_read_time_ms;
        delete[] m_step_reader_idle_time_ms;
        delete[] m_step_size_read_time_ms;
        delete[] m_step_between_read_time_ms;
        delete[] m_step_deser_time_ms;
    }
#endif
#ifdef COUNT_LOST_FRAMES        
    if (chunk_id-1 == m_total_chunks-1)
    {
        printf("total_lost_frames : %u\n",m_lost_frames_count);
        printf("lost frame ids : ");
        for(unsigned int lost_ids : m_lost_frames_id) 
            printf(" %lu ", lost_ids);
        printf("\n");
    }
#endif /* COUNT_LOST_FRAMES */
    return chunks;
}

DataSpacesReader::~DataSpacesReader()
{
    MPI_Barrier(m_gcomm);
    // printf("dspaces_peers : %d\n", dspaces_peers());
    // dspaces_kill(); //This would kill dataspaces_server before other dspaces clients finalize
    dspaces_finalize();
    printf("---===== Finalized dspaces client in DataSpacesReader\n");
}
