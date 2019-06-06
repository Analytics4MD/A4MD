#include "dataspaces_reader.h"
#include "dataspaces.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/serialization/vector.hpp>
#include <sstream>
//#include <TAU.h>

DataSpacesReader::DataSpacesReader(char* var_name, unsigned long int total_chunks, MPI_Comm comm)
: m_size_var_name("chunk_size"),
  m_var_name(var_name),
  m_total_chunks(total_chunks)
  //m_total_data_read_time_ms(0.0),
  //m_total_chunk_read_time_ms(0.0),
  //m_total_reader_idle_time_ms(0.0)
{
    //m_step_chunk_read_time_ms = new double [m_total_chunks];
    //m_step_reader_idle_time_ms = new double [m_total_chunks];
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
    //TimeVar t_start = timeNow();
    unsigned long int chunk_id;
    //printf("----======= Entering DataSpacesReader::get_chunks with chunksfrom %i, chunksto %i\n",chunks_from, chunks_to);
    std::vector<Chunk*> chunks; 
    MPI_Barrier(m_gcomm);
    int ndim = 1;
    uint64_t lb[1] = {0}, ub[1] = {0};
    for (chunk_id = chunks_from; chunk_id<=chunks_to; chunk_id++)
    {
        std::size_t chunk_size;
        //TimeVar t_istart = timeNow();
        //TAU_STATIC_TIMER_START("total_read_idle_time");
        //TAU_DYNAMIC_TIMER_START("read_idle_time");
        dspaces_lock_on_read("size_lock", &m_gcomm);
        //TAU_DYNAMIC_TIMER_STOP("read_idle_time");
        //TAU_STATIC_TIMER_STOP("total_read_idle_time");
        //DurationMilli reader_idle_time_ms = timeNow()-t_istart;
        //m_step_reader_idle_time_ms[chunk_id] = reader_idle_time_ms.count();
        //m_total_reader_idle_time_ms += m_step_reader_idle_time_ms[chunk_id];
        //printf("---==== Reading chunk id %u\n",chunk_id);
        //TimeVar t_rstart = timeNow();
        //TAU_STATIC_TIMER_START("total_read_size_time");
        //TAU_DYNAMIC_TIMER_START("read_size_time");
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
        //TAU_DYNAMIC_TIMER_STOP("read_size_time");
        //TAU_STATIC_TIMER_STOP("total_read_size_time");
        dspaces_unlock_on_read("size_lock", &m_gcomm);
        //printf("chunk size read from ds for chunkid %i : %u\n", chunk_id, chunk_size);

        char *input_data = new char [chunk_size];
        //TAU_STATIC_TIMER_START("total_between_read_time");
        //TAU_DYNAMIC_TIMER_START("between_read_time");
        dspaces_lock_on_read("my_test_lock", &m_gcomm);
        //TAU_DYNAMIC_TIMER_STOP("between_read_time");
        //TAU_STATIC_TIMER_STOP("total_between_read_time");
        
        //TAU_STATIC_TIMER_START("total_read_chunk_time");
        //TAU_DYNAMIC_TIMER_START("read_chunk_time");
        //TAU_TRACK_MEMORY_FOOTPRINT();
        //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
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
        
        //TAU_DYNAMIC_TIMER_STOP("read_chunk_time");
        //TAU_STATIC_TIMER_STOP("total_read_chunk_time");
        //DurationMilli read_chunk_time_ms = timeNow()-t_rstart;
        //m_step_chunk_read_time_ms[chunk_id] = read_chunk_time_ms.count();
        //m_total_chunk_read_time_ms += m_step_chunk_read_time_ms[chunk_id];
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
    //DurationMilli read_time_ms = timeNow()-t_start;
    //m_total_data_read_time_ms += read_time_ms.count();
    //if (chunk_id-1 == m_total_chunks-1)
    //{
    //    printf("total_data_read_time_ms : %f\n",m_total_data_read_time_ms);
    //    printf("total_chunk_read_time_ms : %f\n",m_total_chunk_read_time_ms);
    //    printf("total_reader_idle_time_ms : %f\n",m_total_reader_idle_time_ms);
    //    printf("total_chunks read : %u\n",m_total_chunks);
    //    printf("step_chunk_read_time_ms : ");
    //    for (auto step = 0; step < m_total_chunks; step++)
    //    {
    //        printf(" %f ", m_step_chunk_read_time_ms[step]);
    //    }
    //    printf("\n");
    //    printf("step_reader_idle_time_ms : ");
    //    for (auto step = 0; step < m_total_chunks; step++)
    //    {
    //        printf(" %f ", m_step_reader_idle_time_ms[step]);
    //    }
    //    printf("\n");
    //    //ToDo: delete in destructor
    //    delete[] m_step_chunk_read_time_ms;
    //    delete[] m_step_reader_idle_time_ms;
    //}

    return chunks;
}

DataSpacesReader::~DataSpacesReader()
{
    MPI_Barrier(m_gcomm);
    dspaces_finalize();
    printf("---===== Finalized dspaces client in DataSpacesReader\n");
}
