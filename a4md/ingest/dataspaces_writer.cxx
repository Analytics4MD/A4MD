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

void DataSpacesWriter::write_chunks(PlumedChunker chunker)
{
    MPI_Barrier(m_gcomm);

    std::ostringstream oss;
    {
        boost::archive::text_oarchive oa(oss);
        // write class instance to archive
        oa << chunker;
    }
    std::cout << "Serialized~!!!!!!" << oss.str()  << "\n";

    
    dspaces_lock_on_write("my_test_lock", &m_gcomm);
    //for (auto it = chunks.begin(); it!=chunks.end(); ++it)
    //{
    //    auto chunk = *it;
    //    printf("-== Chunks id %i ==-\n",chunk.chunk_id);
    //    printf("-== Chunk data %s ==-\n",chunk.data);
    int ndim = 1;
    int lb = 0;
    int ub = 0;
    unsigned int ts = 1;
    auto data = oss.str();
    //bool isSuccess = dspaces_put(m_var_name.c_str(),
    //                        ts,
    //                        oss.str().length(),
    //                        ndim,
    //                        lb,
    //                        ub,
    //                        reinterpret_cast<void*>(data));
    PlumedChunker chunksback; 
    std::istringstream iss(oss.str());
    {
        boost::archive::text_iarchive ia(iss);
        ia >> chunksback;
    }

    {
        std::ostringstream oss2;

        boost::archive::text_oarchive oa(oss2);
        oa << chunksback;

        std::cout << oss.str()  << "\n";
        std::cout << oss2.str() << "\n";
    }
    //for (auto i:chunksback)
    //{
    //    printf("In Chunks iTerator\n");
    chunksback.print();
    //}
    //  dspaces_unlock_on_write("my_test_lock", &m_gcomm);
    MPI_Barrier(m_gcomm);
}

void DataSpacesWriter::write_chunks(std::vector<Chunk> chunks)
{
    MPI_Barrier(m_gcomm);

    std::ostringstream oss;
    {
        boost::archive::text_oarchive oa(oss);
        // write class instance to archive
        oa << chunks;
    }

    
    std::cout << "Serialized~!!!!!!" << oss.str()  << "\n";

    std::vector<PLMDChunk> chunksback; 
    std::istringstream iss(oss.str());
    {
        boost::archive::text_iarchive ia(iss);
        ia >> chunksback;
    }

    {
        std::ostringstream oss2;

        boost::archive::text_oarchive oa(oss2);
        oa << chunksback;

        std::cout << oss.str()  << "\n";
        std::cout << oss2.str() << "\n";
    }
    for (auto i:chunksback)
    {
        printf("In Chunks iTerator\n");
        i.print();
    }
    //  dspaces_unlock_on_write("my_test_lock", &m_gcomm);
    MPI_Barrier(m_gcomm);
}
