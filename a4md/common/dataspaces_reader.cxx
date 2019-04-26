#include "dataspaces_reader.h"
#include "dataspaces.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/serialization/vector.hpp>
#include <sstream>
#include <chrono>
#include <thread>


DataSpacesReader::DataSpacesReader(char* var_name, unsigned long int total_chunks, MPI_Comm comm, bool count_lost_frames)
: m_size_var_name("chunk_size"),
  m_var_name(var_name),
  m_total_chunks(total_chunks),
  m_total_data_read_time_ms(0.0),
  m_total_chunk_read_time_ms(0.0),
  m_total_reader_idle_time_ms(0.0),
  m_wait_ms(1000),// default wait time of 1 second
  m_min_wait_ms(100),// default min wait time of 100 ms
  m_max_wait_ms(30000),//defailt max wait time of 30 seconds
  m_count_lost_frames(count_lost_frames),
  m_lost_frames_count(0)	
{
    m_step_chunk_read_time_ms = new double [m_total_chunks];
    m_step_reader_idle_time_ms = new double [m_total_chunks];
    m_step_size_read_time_ms = new double [m_total_chunks];
    m_step_between_read_time_ms = new double [m_total_chunks];
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
    printf("Initialized dspaces client in DataSpacesReader, var_name : %s, total_chunks: %lu \n", var_name, total_chunks);
}

std::vector<Chunk*> DataSpacesReader::get_chunks(unsigned long int chunks_from, unsigned long int chunks_to)
{
    TimeVar t_start = timeNow();
    unsigned long int chunk_id;
    //printf("----======= Entering DataSpacesReader::get_chunks with chunksfrom %lu, chunksto %lu\n",chunks_from, chunks_to);
    std::vector<Chunk*> chunks; 
    MPI_Barrier(m_gcomm);
    int ndim = 1;
    uint64_t lb[1] = {0}, ub[1] = {0};
    for (chunk_id = chunks_from; chunk_id<=chunks_to; chunk_id++)
    {
	    if (m_count_lost_frames)
	    {
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
            	        printf("Dont have chunk %lu yet. So waiting %lu ms to poll\n",chunk_id,m_wait_ms);
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
                continue;
            }
        }
	
        std::size_t chunk_size;
        TimeVar t_istart = timeNow();
        dspaces_lock_on_read("size_lock", &m_gcomm);
        DurationMilli reader_idle_time_ms = timeNow()-t_istart;
        m_step_reader_idle_time_ms[chunk_id] = reader_idle_time_ms.count();
        m_total_reader_idle_time_ms += m_step_reader_idle_time_ms[chunk_id];
        //printf("---==== Reading chunk id %u\n",chunk_id);
        TimeVar t_rsstart = timeNow();
        int error = dspaces_get(m_size_var_name.c_str(),
                                chunk_id,
                                sizeof(std::size_t),
                                ndim,
                                lb,
                                ub,
                                &chunk_size);

        if (error != 0)
            printf("----====== ERROR (%i): Did not read SIZE of chunk id: %lu from dataspaces successfully\n",error, chunk_id); 
        if (error == -11)
        {
            printf("Recieved -11 from dspaces get. Probably lost chunk %lu\n",chunk_id);
            if (m_count_lost_frames)
            {
                m_lost_frames_count++;
                continue;
            }
            else
            {
                throw new DataLayerException("Dataspaces get recieved error code -11. This is not expected for lock type 2, but expected for lock type 1 or 3. Check lock type used.\n");
            }
        }
        //    printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        //else
        DurationMilli size_read_time_ms = timeNow() - t_rsstart;
        m_step_size_read_time_ms[chunk_id] = size_read_time_ms.count();
        dspaces_unlock_on_read("size_lock", &m_gcomm);
        //printf("chunk size read from ds for chunkid %i : %u\n", chunk_id, chunk_size);

        char *input_data = new char [chunk_size];

        TimeVar t_rbstart = timeNow();
        dspaces_lock_on_read("my_test_lock", &m_gcomm);
        DurationMilli between_read_time_ms = timeNow() - t_rbstart;
        m_step_between_read_time_ms[chunk_id] = between_read_time_ms.count();
        
        TimeVar t_rcstart = timeNow();
        error = dspaces_get(m_var_name.c_str(),
                            chunk_id,
                            chunk_size,
                            ndim,
                            lb,
                            ub,
                            input_data);

        if (error != 0)
            printf("----====== ERROR (%i): Did not read chunkid %lu from dataspaces successfully\n",error, chunk_id);
        //else
        //    printf("Read chunk id %i from dataspacess successfull\n",chunk_id);
        if (error == -11)
        {
            printf("Recieved -11 from dspaces get. Probably lost chunk %lu\n",chunk_id);
            if (m_count_lost_frames)
            {
                m_lost_frames_count++;
                continue;
            }
            else
            {
                throw new DataLayerException("Dataspaces get recieved error code -11. This is not expected for lock type 2, but expected for lock type 1 or 3. Check lock type used.\n");
            }
        }
                
        DurationMilli read_chunk_time_ms = timeNow()-t_rcstart;
        m_step_chunk_read_time_ms[chunk_id] = read_chunk_time_ms.count();
        m_total_chunk_read_time_ms += m_step_chunk_read_time_ms[chunk_id];
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
    }
    //MPI_Barrier(m_gcomm);
    DurationMilli read_time_ms = timeNow()-t_start;
    m_total_data_read_time_ms += read_time_ms.count();
    if (chunk_id-1 == m_total_chunks-1)
    {
        printf("total_data_read_time_ms : %f\n",m_total_data_read_time_ms);
        printf("total_chunk_read_time_ms : %f\n",m_total_chunk_read_time_ms);
        printf("total_reader_idle_time_ms : %f\n",m_total_reader_idle_time_ms);
        printf("total_chunks read : %u\n",m_total_chunks);
        printf("total_lost_frames : %u\n",m_lost_frames_count);
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

    return chunks;
}

DataSpacesReader::~DataSpacesReader()
{
    MPI_Barrier(m_gcomm);
    dspaces_finalize();
    printf("Finalized dspaces client in DataSpacesReader\n");
}
