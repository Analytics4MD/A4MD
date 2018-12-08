#include "dataspaces_reader.h"
#include "dataspaces.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/serialization/vector.hpp>
#include <sstream>


DataSpacesReader::DataSpacesReader(char* var_name)
{
    MPI_Barrier(MPI_COMM_WORLD);
    m_gcomm = MPI_COMM_WORLD;
    // Initalize DataSpaces
    // # of Peers, Application ID, ptr MPI comm, additional parameters
    // # Peers: Number of connecting clients to the DS server
    // Application ID: Unique idenitifier (integer) for application
    // Pointer to the MPI Communicator, allows DS Layer to use MPI barrier func
    // Addt'l parameters: Placeholder for future arguments, currently NULL.
    dspaces_init(1, 2, &m_gcomm, NULL);
    printf("Initialized dspaces client in DataSpacesReader\n");
    m_var_name = var_name;
}

std::vector<Chunk*> DataSpacesReader::get_chunks(int chunks_from, int chunks_to)
{
    //printf("----======= Entering DataSpacesReader::get_chunks with chunksfrom %i, chunksto %i\n",chunks_from, chunks_to);
    ChunkArray chunks; 
    MPI_Barrier(m_gcomm);
    int ndim = 1;
    uint64_t lb[1] = {0}, ub[1] = {0};
    unsigned int ts = chunks_to;//chunks.get_chunk_id();

    std::string::size_type chunk_size;

    std::string size_var_name = "chunk_size";
    dspaces_lock_on_read("size_lock", &m_gcomm);
    int error = dspaces_get(size_var_name.c_str(),
                            ts,
                            sizeof(std::string::size_type),
                            ndim,
                            lb,
                            ub,
                            &chunk_size);
    if (error != 0)
        printf("----====== ERROR (%i): Did not read SIZE of chunk id: %i from dataspaces successfully\n",error, ts);
    //    printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), ts);
    //else
    dspaces_unlock_on_read("size_lock", &m_gcomm);
    //printf("chunk size read from ds for chunkid %i : %u\n", ts, chunk_size);

    char input_data[chunk_size];

    dspaces_lock_on_read("my_test_lock", &m_gcomm);
    error = dspaces_get(m_var_name.c_str(),
                        ts,
                        chunk_size,
                        ndim,
                        lb,
                        ub,
                        input_data);

    if (error != 0)
        printf("----====== ERROR (%i): Did not read chunkid %i from dataspaces successfully\n",error, ts);
    //else
    //    printf("Read chunk id %i from dataspacess successfull\n",ts);
    
    dspaces_unlock_on_read("my_test_lock", &m_gcomm);

    //printf("Read char array from dataspace:\n %s\n",input_data);
    std::string instr(input_data);
    std::istringstream iss(instr);//oss.str());
    {
        boost::archive::text_iarchive ia(iss);
        ia >> chunks;
    }
  
    //printf("----===== Read chunk array ");
    //chunks.print();
    MPI_Barrier(m_gcomm);
    return chunks.get_chunks();
}

