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


DataSpacesWriter::DataSpacesWriter(char* var_name, unsigned long int total_chunks, MPI_Comm comm)
: m_size_var_name("chunk_size"),
  m_var_name(var_name),
  m_total_chunks(total_chunks),
  m_total_data_write_time_ms(0.0),
  m_total_chunk_write_time_ms(0.0),
  m_total_writer_idle_time_ms(0.0)
{
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
    dspaces_init(nprocs, 1, &m_gcomm, NULL);
    printf("Initialized dspaces client in DataSpacesWriter, var_name: %s, total_chunks: %u\n",m_var_name.c_str(), m_total_chunks);
}

static inline std::size_t round_up_8(std::size_t n)
{
    return (n%8 == 0) ? n : (n/8 + 1)*8;
}

void DataSpacesWriter::write_chunks(std::vector<Chunk*> chunks)
{
    TimeVar t_start = timeNow();
    unsigned long int chunk_id; 
    MPI_Barrier(m_gcomm);
    //printf("Printing chunk before serializing\n");
    for (auto chunk:chunks)
    {
        //chk_ary.print();

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
        printf("chunk size for chunk_id %i is %zu\n",chunk_id,size);
	m_total_size += size;
	TimeVar t_wstart = timeNow();
        dspaces_lock_on_write("size_lock", &m_gcomm);
        DurationMilli writer_idle_time_ms = timeNow()-t_wstart;
	m_total_writer_idle_time_ms += writer_idle_time_ms.count();
        int error = dspaces_put(m_size_var_name.c_str(),
                                chunk_id,
                                sizeof(std::string::size_type),
                                ndim,
                                lb,
                                ub,
				&size);

       
        if (error != 0)
            printf("----====== ERROR: Did not write size of chunk id: %i to dataspaces successfully\n",chunk_id);
        //else
        //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
              
	dspaces_unlock_on_write("size_lock", &m_gcomm);
        //printf("writing char array to dataspace:\n %s\n",data.c_str());
        dspaces_lock_on_write("my_test_lock", &m_gcomm);
        error = dspaces_put(m_var_name.c_str(),
                            chunk_id,
                            data.length(),
                            ndim,
                            lb,
                            ub,
			    data.c_str());
        if (error != 0)
            printf("----====== ERROR: Did not write chunk id: %i to dataspaces successfully\n",chunk_id);
        //else
        //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        dspaces_unlock_on_write("my_test_lock", &m_gcomm);
        DurationMilli write_chunk_time_ms = timeNow()-t_wstart;
        m_total_chunk_write_time_ms += write_chunk_time_ms.count();
    }
    MPI_Barrier(m_gcomm);
    DurationMilli write_time_ms = timeNow()-t_start;
    m_total_data_write_time_ms += write_time_ms.count();
    if (chunk_id == m_total_chunks-1)
    {
        printf("total_data_write_time_ms : %f\n",m_total_data_write_time_ms);
        printf("total_chunk_write_time_ms : %f\n",m_total_chunk_write_time_ms);
	printf("total_writer_idle_time_ms : %f\n",m_total_writer_idle_time_ms);
        printf("total_chunk_data_written : %u\n",m_total_size);
        printf("total_chunks written : %u\n",m_total_chunks);
    }
}
