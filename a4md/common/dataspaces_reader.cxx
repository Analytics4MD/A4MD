#include "dataspaces_reader.h"
#include "dataspaces.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/serialization/vector.hpp>
#include <sstream>


DataSpacesReader::DataSpacesReader(char* var_name, unsigned long int total_chunks, MPI_Comm comm)
: m_size_var_name("chunk_size"),
  m_var_name(var_name),
  m_total_chunks(total_chunks),
  m_total_data_read_time_ms(0.0),
  m_total_chunk_read_time_ms(0.0),
  m_total_reader_idle_time_ms(0.0)
{
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
    printf("Initialized dspaces client in DataSpacesReader, var_name : %s, total_chunks: %u \n", var_name, total_chunks);
}

std::vector<Chunk*> DataSpacesReader::get_chunks(unsigned long int chunks_from, unsigned long int chunks_to)
{
    TimeVar t_start = timeNow();
    unsigned long int chunk_id;
    //printf("----======= Entering DataSpacesReader::get_chunks with chunksfrom %i, chunksto %i\n",chunks_from, chunks_to);
    std::vector<Chunk*> chunks; 
    MPI_Barrier(m_gcomm);
    int ndim = 1;
    uint64_t lb[1] = {0}, ub[1] = {0};
    for (chunk_id = chunks_from; chunk_id<=chunks_to; chunk_id++)
    {
        std::string::size_type chunk_size;
        TimeVar t_rstart = timeNow();
        dspaces_lock_on_read("size_lock", &m_gcomm);
	DurationMilli reader_idle_time_ms = timeNow()-t_rstart;
        m_total_reader_idle_time_ms += reader_idle_time_ms.count();
        //printf("---==== Reading chunk id %u\n",chunk_id);
        int error = dspaces_get(m_size_var_name.c_str(),
                                chunk_id,
                                sizeof(std::string::size_type),
                                ndim,
                                lb,
                                ub,
                                &chunk_size);
        if (error != 0)
            printf("----====== ERROR (%i): Did not read SIZE of chunk id: %i from dataspaces successfully\n",error, chunk_id);
        //    printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        //else
        dspaces_unlock_on_read("size_lock", &m_gcomm);
        //printf("chunk size read from ds for chunkid %i : %u\n", chunk_id, chunk_size);

        char input_data[chunk_size];

        dspaces_lock_on_read("my_test_lock", &m_gcomm);
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
        
        dspaces_unlock_on_read("my_test_lock", &m_gcomm);
        DurationMilli read_chunk_time_ms = timeNow()-t_rstart;
        m_total_chunk_read_time_ms += read_chunk_time_ms.count();

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
    MPI_Barrier(m_gcomm);
    DurationMilli read_time_ms = timeNow()-t_start;
    m_total_data_read_time_ms += read_time_ms.count();
    if (chunk_id-1 == m_total_chunks-1)
    {
        printf("total_data_read_time_ms : %f\n",m_total_data_read_time_ms);
        printf("total_chunk_read_time_ms : %f\n",m_total_chunk_read_time_ms);
	printf("total_reader_idle_time_ms : %f\n",m_total_reader_idle_time_ms);
        printf("total_chunks read : %u\n",m_total_chunks);
    }

    return chunks;
}

