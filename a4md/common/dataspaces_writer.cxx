#include "dataspaces_writer.h"
#include "dataspaces.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/serialization/vector.hpp>
#include <sstream>


DataSpacesWriter::DataSpacesWriter(char* var_name)
{
    MPI_Barrier(MPI_COMM_WORLD);
    m_gcomm = MPI_COMM_WORLD;
    // Initalize DataSpaces
    // # of Peers, Application ID, ptr MPI comm, additional parameters
    // # Peers: Number of connecting clients to the DS server
    // Application ID: Unique idenitifier (integer) for application
    // Pointer to the MPI Communicator, allows DS Layer to use MPI barrier func
    // Addt'l parameters: Placeholder for future arguments, currently NULL.
    dspaces_init(1, 1, &m_gcomm, NULL);
    printf("Initialized dspaces client in DataSpacesWriter\n");
    m_var_name = var_name;
}

void DataSpacesWriter::write_chunks(std::vector<Chunk*> chunks)
{
    MPI_Barrier(m_gcomm);
    //printf("Printing chunk before serializing\n");
    ChunkArray chk_ary;
    for (auto ichunk:chunks)
        chk_ary.append(ichunk);
    //chk_ary.print();


    std::ostringstream oss;
    {
        boost::archive::text_oarchive oa(oss);
        // write class instance to archive
        oa << chk_ary;
    }
    
    //ChunkArray inchunks;
    //std::string instr(oss.str());
    //std::istringstream iss(instr);//oss.str());
    //{
    //    boost::archive::text_iarchive ia(iss);
    //    ia >> inchunks;
    //}
    //printf("Printing chunk after serializing\n");
    //inchunks.print();
 
    int ndim = 1;
    uint64_t lb[1] = {0}, ub[1] = {0};
    unsigned int ts = chk_ary.get_last_chunk_id();
    std::string data = oss.str();
    std::string::size_type size = data.length();

    std::string size_var_name = "chunk_size";
    printf("chunk size for ts %i is %i\n",ts,size);
    dspaces_lock_on_write("size_lock", &m_gcomm);
    int error = dspaces_put(size_var_name.c_str(),
                            ts,
                            1*sizeof(std::string::size_type),
                            ndim,
                            lb,
                            ub,
                            &size);
    if (error != 0)
        printf("----====== ERROR: Did not write size of chunk id: %i to dataspaces successfully\n",ts);
    //else
    //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), ts);
    dspaces_unlock_on_write("size_lock", &m_gcomm);

    //printf("writing char array to dataspace:\n %s\n",data.c_str());

    dspaces_lock_on_write("my_test_lock", &m_gcomm);
    error = dspaces_put(m_var_name.c_str(),
                        ts,
                        data.length(),
                        ndim,
                        lb,
                        ub,
                        data.c_str());
    if (error != 0)
        printf("----====== ERROR: Did not write chunk id: %i to dataspaces successfully\n",ts);
    //else
    //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), ts);
    dspaces_unlock_on_write("my_test_lock", &m_gcomm);

    MPI_Barrier(m_gcomm);
}
